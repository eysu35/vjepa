# inference.py
import os
import sys
import argparse
import yaml
import time
import copy
import tempfile # For temporary CSV
import csv      # For writing temporary CSV

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.models.vision_transformer import VisionTransformer # Assuming 
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_video_model # Use this helper if convenient
from src.datasets.data_manager import init_data

from src.utils.logging import get_logger
from src.utils.tensors import repeat_interleave_batch
import torchvision.transforms.functional as TF # For visualization

logger = get_logger(__name__)

# --- Visualization Helper ---
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverses normalization for viewing."""
    # Ensure tensor is detached and on CPU
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4: # Handle batch dimension if present
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Input tensor must have 3 dimensions (C, H, W), got {tensor.ndim}")

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return TF.to_pil_image(tensor)
def main_inference(args):
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    cfgs_meta = config.get('meta', {})
    cfgs_mask = config.get('mask', {})
    cfgs_model = config.get('model', {})
    cfgs_data = config.get('data', {})
    cfgs_data_aug = config.get('data_aug', {})

    which_dtype = cfgs_meta.get('dtype', 'float32')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(which_dtype.lower(), torch.float32)
    logger.info(f"Using data type: {dtype}")

    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    num_frames = cfgs_data.get('num_frames') # Frames model expects per clip
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate', 4) # Get sampling rate from config
    duration = cfgs_data.get('clip_duration') # Get duration if specified
    num_clips = cfgs_data.get('num_clips', 1) # How many clips per video instance (often 1 for VJEPA)
    decode_one_clip = cfgs_data.get('decode_one_clip', True) # Common setting
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset') # Usually 'videodataset' or 'webdataset'
    pin_mem = cfgs_data.get('pin_mem', False)

    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_sdpa = cfgs_meta.get('use_sdpa', False)

    # Data Aug settings (for transform consistency, but we'll disable randomness)
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [1.0, 1.0]) # Fixed AR for inference
    rr_scale = cfgs_data_aug.get('random_resize_scale', [1.0, 1.0]) # Fixed scale for inference
    motion_shift = False # Disable for inference
    reprob = 0.0 # Disable for inference
    use_aa = False # Disable for inference

    # --- 2. Setup Device ---
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- 3. Initialize Model ---
    logger.info(f"Initializing model: {model_name}")
    encoder, predictor = init_video_model(
        uniform_power=uniform_power, use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask), zero_init_mask_tokens=zero_init_mask_tokens,
        device='cpu', patch_size=patch_size, num_frames=num_frames, tubelet_size=tubelet_size,
        model_name=model_name, crop_size=crop_size, pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim, use_sdpa=use_sdpa)
    target_encoder = copy.deepcopy(encoder)

    # --- 4. Load Pretrained Weights (Using .pth.tar logic from previous answer) ---
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint dictionary keys found: {list(checkpoint.keys())}")

        if 'encoder' not in checkpoint: raise KeyError("Checkpoint missing 'encoder' state_dict.")
        if 'predictor' not in checkpoint: raise KeyError("Checkpoint missing 'predictor' state_dict.")
        target_encoder_state_dict_loaded = checkpoint.get('target_encoder', checkpoint['encoder'])

        encoder_state_dict_loaded = checkpoint['encoder']
        predictor_state_dict_loaded = checkpoint['predictor']

        encoder_state_dict_cleaned = {k.replace('module.', '', 1): v for k, v in encoder_state_dict_loaded.items()}
        predictor_state_dict_cleaned = {k.replace('module.', '', 1): v for k, v in predictor_state_dict_loaded.items()}
        target_encoder_state_dict_cleaned = {k.replace('module.', '', 1): v for k, v in target_encoder_state_dict_loaded.items()}

        missing_keys_enc, _ = encoder.load_state_dict(encoder_state_dict_cleaned, strict=False)
        missing_keys_pred, _ = predictor.load_state_dict(predictor_state_dict_cleaned, strict=False)
        missing_keys_target, _ = target_encoder.load_state_dict(target_encoder_state_dict_cleaned, strict=False)
        logger.info(f"Encoder missing keys: {len(missing_keys_enc)}")
        logger.info(f"Predictor missing keys: {len(missing_keys_pred)}")
        logger.info(f"Target Encoder missing keys: {len(missing_keys_target)}")

    except Exception as e:
        logger.error(f"Error loading checkpoint weights from {args.checkpoint_path}: {e}", exc_info=True)
        sys.exit(1)

    encoder.to(device).eval()
    predictor.to(device).eval()
    target_encoder.to(device).eval()
    for param in target_encoder.parameters(): param.requires_grad = False
    logger.info("Models initialized and weights loaded successfully.")

    # --- 5. Prepare Transforms and Mask Collator ---
    # Use transforms consistent with training, but disable randomness for inference
    transform = make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale, reprob=reprob, auto_augment=use_aa,
        motion_shift=motion_shift, crop_size=crop_size)
    logger.info("Transforms prepared.")

    # Initialize Mask Collator
    MaskCollatorClass = MB3DMaskCollator if mask_type == 'multiblock3d' else TubeMaskCollator
    mask_collator = MaskCollatorClass(
        crop_size=crop_size, num_frames=num_frames, patch_size=patch_size,
        tubelet_size=tubelet_size, cfgs_mask=cfgs_mask)
    logger.info(f"Mask generator ({mask_type}) prepared.")

    # --- 6. Setup DataLoader for Single Video ---
    logger.info(f"Setting up DataLoader for single video: {args.video_path}")
    video_abs_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_abs_path):
        logger.error(f"Video file not found at resolved path: {video_abs_path}")
        sys.exit(1)

    # Create a temporary CSV file
    temp_csv_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv_file = f.name
            writer = csv.writer(f)
            # Format: /path/to/video.mp4 label (label is ignored but needed)
            writer.writerow([video_abs_path, 0])
        logger.info(f"Created temporary dataset CSV: {temp_csv_file}")

        # Initialize DataLoader using the temporary CSV
        # Ensure world_size=1 and rank=0 for non-distributed enference
        video_abs_path = [video_abs_path]
        logger.info(f"created video path list: {video_abs_path}")
           
        (dataloader, _) = init_data(
            data=dataset_type,
            root_path=video_abs_path, # Use the temporary CSV
            batch_size=1,            # Process one video at a time
            training=False,          # Use evaluation mode transforms/sampling if different
            clip_len=num_frames,
            frame_sample_rate=sampling_rate,
            filter_short_videos=False, # Load even if slightly shorter than expected
            decode_one_clip=decode_one_clip,
            duration=duration,
            num_clips=num_clips,
            transform=transform,     # Apply the same transforms
            datasets_weights=None,   # Not needed for single file
            collator=mask_collator,  # Use the same mask collator
            num_workers=args.num_workers, # Can set low for inference
            world_size=1,            # Single process
            pin_mem=pin_mem,
            rank=0,                  # Single process rank
            log_dir=None             # No resource logging needed
        )
        logger.info("DataLoader initialized for the video.")

        # --- 7. Load Data and Masks from DataLoader ---
        logger.info("Loading data and generating masks via DataLoader...")
        
        dataloader_iter = iter(dataloader)
        # Fetch the single batch
        loaded_batch = next(dataloader_iter)

        # Extract data and masks (structure depends on the specific collator output)
        # Common structure: (list_of_clip_tensors, masks_enc_list, masks_pred_list), Optional(labels)
        # Assuming first element is the list of clip tensors (should be just one tensor for B=1, num_clips=1)
        breakpoint()
        video_tensor_list = loaded_batch[0]
        masks_enc = loaded_batch[1] # List of encoder masks
        masks_pred = loaded_batch[2] # List of predictor masks

        if not isinstance(video_tensor_list, list) or len(video_tensor_list) != 1:
             logger.warning(f"Unexpected video tensor format from loader: {type(video_tensor_list)}. Expected list of length 1.")
             # Attempt to handle if it's just the tensor directly (B, T, C, H, W)
             if isinstance(video_tensor_list, torch.Tensor) and video_tensor_list.ndim == 5:
                 video_tensor = video_tensor_list
             else:
                  raise ValueError("Cannot extract video tensor from DataLoader output.")
        else:
            video_tensor = video_tensor_list[0] # Shape [B, T, C, H, W], B should be 1

        # Ensure tensor and masks are on the correct device
        video_tensor = video_tensor.to(device)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        logger.info(f"Data loaded. Video tensor shape: {video_tensor.shape}") # Should be [1, num_frames, C, H, W]
        logger.info(f"Generated {len(masks_enc)} encoder masks and {len(masks_pred)} predictor masks.")

    except StopIteration:
        logger.error("DataLoader failed to yield any data. Check video file, paths, and dataset/loader settings.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during data loading: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up the temporary CSV file
        if temp_csv_file and os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)
            logger.info(f"Removed temporary dataset CSV: {temp_csv_file}")


    # --- 8. Perform Inference ---
    logger.info("Running model inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(dtype != torch.float32)):
            # Step 1: Get target features 'h'
            h = target_encoder(video_tensor)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred, concat=False) # List of [B, N_pred, D]

            # Step 2: Get context features 'z_context'
            z_context = encoder(video_tensor, masks_enc) # [B, N_ctx, D]

            # Step 3: Get latent predictions 'z'
            z = predictor(z_context, h, masks_enc, masks_pred) # List of [B, N_pred, D]

    logger.info(f"Inference complete. Got {len(z)} latent prediction tensors.")

    # --- 9. Visualization ---
    logger.info("Preparing visualization...")
    num_vis_frames = min(args.num_vis, num_frames)

    # Get original frames for display (unnormalize the input tensor)
    # Squeeze batch dimension before unnormalizing
    original_frames_display = [unnormalize(video_tensor[0, i]) for i in range(num_vis_frames)]

    # Create masked input visualization (using placeholder/approximate logic as before)
    masked_input_frames_display = []
    try:
        patches_per_frame = (crop_size // patch_size) ** 2
        num_temporal_patches = num_frames // tubelet_size
        total_patches = patches_per_frame * num_temporal_patches
        # Use first encoder mask to estimate ratio (adapt if mask structure is different)
        # masks_enc[0] might be indices or boolean mask. Assuming indices shape [B, 1, N_kept]
        mask_ratio = 1.0 - (masks_enc[0].shape[-1] / total_patches) if (masks_enc and masks_enc[0] is not None and total_patches > 0) else 0.5
        logger.info(f"Approximate visual masking ratio: {mask_ratio:.2f}")

        for frame_pil in original_frames_display:
            frame_np = np.array(frame_pil).copy()
            h, w, _ = frame_np.shape
            ph, pw = patch_size, patch_size
            for r in range(0, h, ph):
                for c in range(0, w, pw):
                    if np.random.rand() < mask_ratio:
                        frame_np[r:r+ph, c:c+pw, :] = 0 # Black out patch
            masked_input_frames_display.append(Image.fromarray(frame_np))
    except Exception as e:
        logger.warning(f"Could not generate masked input visualization accurately: {e}. Using simple overlay.")
        masked_input_frames_display = [img.copy() for img in original_frames_display]


    # --- Predicted Pixels (Requires Decoder - Placeholder) ---
    logger.warning("--------------------------------------------------------------------")
    logger.warning("Model produced LATENT predictions (variable 'z').")
    logger.warning("Visualizing these requires a Pixel Decoder specific to this model.")
    logger.warning("The 'Predicted' row below will show placeholders.")
    logger.warning("--------------------------------------------------------------------")

    # --- Plotting ---
    fig, axes = plt.subplots(3, num_vis_frames, figsize=(num_vis_frames * 2, 6))
    fig.suptitle(f"VJEPA Inference: {os.path.basename(args.video_path)}")

    for i in range(num_vis_frames):
        ax = axes[0, i] # Original
        if i < len(original_frames_display): ax.imshow(original_frames_display[i])
        ax.set_title(f"Frame {i}")
        ax.axis('off')

        ax = axes[1, i] # Masked Input
        if i < len(masked_input_frames_display): ax.imshow(masked_input_frames_display[i])
        ax.set_title(f"Masked In {i}")
        ax.axis('off')

        ax = axes[2, i] # Predicted (Placeholder)
        placeholder_img = Image.new('RGB', original_frames_display[0].size, (50, 50, 50))
        ax.imshow(placeholder_img)
        ax.set_title(f"Pred {i} (N/A)")
        ax.axis('off')

    axes[0, 0].set_ylabel("Original", rotation=90, labelpad=20, verticalalignment='center', fontsize=10)
    axes[1, 0].set_ylabel("Masked Input\n(Approx.)", rotation=90, labelpad=20, verticalalignment='center', fontsize=10)
    axes[2, 0].set_ylabel("Predicted\n(Requires Decoder)", rotation=90, labelpad=20, verticalalignment='center', fontsize=10)

    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    if args.output_image:
        plt.savefig(args.output_image)
        logger.info(f"Visualization saved to {args.output_image}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VJEPA Inference Script using DataLoader')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file (e.g., .mp4)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the pretrained model checkpoint file (e.g., .pth.tar)')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model/data configuration YAML file')
    parser.add_argument('--output_image', type=str, default=None, help='Path to save the output visualization image (optional)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--num_vis', type=int, default=8, help='Number of frames to visualize in the output image')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU workers for data loading')


    args = parser.parse_args()

    # Basic validation (paths checked later or inside init_data)
    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)
    if not os.path.isfile(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)

    main_inference(args)
