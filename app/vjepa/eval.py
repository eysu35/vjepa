# eval.py
import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import logging 

from src.utils.logging import AverageMeter
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch

logger = logging.getLogger(__name__) # Or use your project's get_logger

@torch.no_grad() # Ensure no gradients are computed during evaluation
def evaluate(
    encoder,
    predictor,
    target_encoder,
    eval_loader,
    mask_collator, # Pass the mask collator/generator object
    device,
    dtype,
    mixed_precision,
    loss_exp,
    num_clips,
    batch_size, # Needed for repeat_interleave_batch inside eval
    world_size, # For distributed averaging
    cfgs_mask, # Needed if mask collator needs re-initialization per batch or similar logic
    rank):
    """
    Performs evaluation on the evaluation dataset.
    Moved from train.py for modularity.
    """
    # Set models to evaluation mode
    encoder.eval()
    predictor.eval()
    target_encoder.eval() # Target encoder is already in no_grad mode effectively, but set eval for consistency

    eval_loss_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    def forward_target(c, _masks_pred):
        h = target_encoder(c)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, _masks_pred, concat=False)
        return h

    def forward_context(c, h, _masks_enc, _masks_pred):
        z = encoder(c, _masks_enc)
        z = predictor(z, h, _masks_enc, _masks_pred)
        return z

    def loss_fn(z, h):
        loss = 0.
        for zi, hi in zip(z, h):
            loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
        loss /= len(masks_pred) 
        return loss

    logger.info(f"[Rank {rank}] Starting evaluation run...")
    eval_loader_iter = iter(eval_loader)
    start_time = time.time()

    for i in range(len(eval_loader)): # Iterate through the eval loader
        batch_start_time = time.time()
        try:
            # --- Get Data and Masks ---
            # This assumes the eval_loader's collate_fn (mask_collator)
            # provides data and masks in the expected format.
            udata, masks_enc, masks_pred = next(eval_loader_iter)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            # --- Load clips and masks to device ---
            clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
            _masks_enc, _masks_pred = [], []
            current_batch_size = udata[0][0].shape[0] # Get actual batch size

            for _me, _mp in zip(masks_enc, masks_pred):
                _me = _me.to(device, non_blocking=True)
                _mp = _mp.to(device, non_blocking=True)
                _me = repeat_interleave_batch(_me, current_batch_size, repeat=num_clips)
                _mp = repeat_interleave_batch(_mp, current_batch_size, repeat=num_clips)
                _masks_enc.append(_me)
                _masks_pred.append(_mp)
            # --- End loading clips ---

            # --- Perform Forward Pass under Autocast and No Grad ---
            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                h = forward_target(clips, _masks_pred)
                z = forward_context(clips, h, _masks_enc, _masks_pred)
                loss = loss_fn(z, h)

            # --- Update Loss Meter ---
            if not torch.isnan(loss) and not torch.isinf(loss):
                eval_loss_meter.update(loss.item())
            else:
                logger.warning(f"[Rank {rank}] Warning: NaN or Inf loss detected during evaluation batch {i}, skipping.")

            batch_time_meter.update(time.time() - batch_start_time)

        except StopIteration:
            logger.warning(f"[Rank {rank}] Eval loader iterator stopped unexpectedly at batch {i}.")
            break # End of loader
        except Exception as e:
            logger.error(f"[Rank {rank}] Error during evaluation batch {i}: {e}", exc_info=True)
            # Decide if you want to skip the batch or stop evaluation
            continue # Skip batch on error

    # --- Calculate Average Loss ---
    avg_loss = eval_loss_meter.avg if eval_loss_meter.count > 0 else float('inf')
    total_eval_time = time.time() - start_time

    # --- Average loss across DDP processes ---
    if dist.is_initialized() and world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    # --- Set models back to training mode ---
    # It's crucial that the caller (train.py) handles this if necessary,
    # but doing it here ensures the models passed back are in train state.
    encoder.train()
    predictor.train()
    # target_encoder remains in no_grad mode

    if eval_loss_meter.count > 0:
         logger.info(f"[Rank {rank}] Finished evaluation. Avg Loss: {avg_loss:.4f} over {eval_loss_meter.count} batches. Avg Batch Time: {batch_time_meter.avg*1000:.1f} ms. Total Time: {total_eval_time:.2f} s")
    else:
         logger.warning(f"[Rank {rank}] Finished evaluation but processed 0 valid batches.")

    return avg_loss, total_eval_time # Return loss and total time taken
