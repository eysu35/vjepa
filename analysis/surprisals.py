import torch
import matplotlib.pyplot as plt
import numpy as np
import os

saycam_file_paths = [
    "vjepa_say_base_no_rope/intuitive_physics/inflevel_lab-default/losses_continuity_5fs_4_8_12_16_20ctxt.pth",
    "vjepa_say_base_no_rope/intuitive_physics/inflevel_lab-default/losses_gravity_5fs_4_8_12_16_20ctxt.pth",
    "vjepa_say_base_no_rope/intuitive_physics/inflevel_lab-default/losses_solidity_5fs_4_8_12_16_20ctxt.pth",
    ]

meta_file_paths = [
    "vit-l-rope-howto-0/inflevel/raw_surprises/continuity_32frames.pth",
    "vit-l-rope-howto-0/inflevel/raw_surprises/gravity_32frames.pth",
    "vit-l-rope-howto-0/inflevel/raw_surprises/solidity_32frames.pth"
]

labels = ["Continuity", "Gravity", "Solidity"]
def load_metrics_data(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path} (CWD: {os.getcwd()})") # Added CWD for context
        return None
    try:
        device = torch.device('cpu')
        loaded_data = torch.load(file_path, map_location=device)
        print(f"Successfully loaded metrics from: {file_path}")
        return loaded_data
    except Exception as e:
        print(f"ERROR: Could not load metrics file '{file_path}': {e}")
        return None

def main():
  save_plot_directory_base = os.path.expanduser("~/Desktop/")

  i = 0
  for saycam_file, meta_file_path in zip(saycam_file_paths, meta_file_paths):

    saycam_metrics = load_metrics_data(saycam_file)
    meta_metrics = load_metrics_data(meta_file_path)

    say_raw_losses_data = saycam_metrics['losses']
    say_losses_np = np.array([item.cpu().numpy() for item in say_raw_losses_data])
    say_losses_for_plot = np.mean(say_losses_np, axis=(0, 1))
    norm_say = np.linalg.norm(say_losses_for_plot)
    say_losses_for_plot = say_losses_for_plot / norm_say if norm_say != 0 else say_losses_for_plot

    meta_raw_losses_data = meta_metrics['losses']
    meta_losses_np = np.array([item.cpu().numpy() for item in meta_raw_losses_data])
    meta_losses_for_plot = np.mean(meta_losses_np, axis=(0, 1))
    norm_meta = np.linalg.norm(meta_losses_for_plot)
    meta_losses_for_plot = meta_losses_for_plot / norm_meta if norm_meta != 0 else meta_losses_for_plot


    frame_step_info = saycam_metrics.get('frame_step')
    num_frames_from_losses = say_losses_np.shape[2]
    frame_numbers = None
    if frame_step_info is not None:
        if isinstance(frame_step_info, (int, float)) and frame_step_info > 0:
            frame_numbers = np.arange(num_frames_from_losses) * frame_step_info
        elif isinstance(frame_step_info, (list, np.ndarray)):
            if len(frame_step_info) == num_frames_from_losses:
                frame_numbers = np.array(frame_step_info)
    if frame_numbers is None:
        frame_numbers = np.arange(num_frames_from_losses)
        if num_frames_from_losses == 1 and say_losses_for_plot.ndim == 1 and len(say_losses_for_plot) == 1 :
            frame_numbers = np.array([0])
    if not (say_losses_for_plot.shape[0] == len(frame_numbers)):
        print(f"ERROR: Mismatch between 'losses_for_plot' and frame numbers.")
        continue
    if say_losses_for_plot.shape[0] == 0:
        print("ERROR: No data points to plot.")
        continue
    
    plt.figure(figsize=(16, 8))
    plt.plot(frame_numbers, say_losses_for_plot, marker='.', linestyle='-', color='blue')
    plt.plot(frame_numbers, meta_losses_for_plot, marker='.', linestyle='-', color='orange')
    
    plt.legend(loc='best')

    plot_title_parts = [f'Model Surprisals vs. Video Frames, {labels[i]}']
    plt.title("\n".join(plot_title_parts), fontsize=24)
    plt.xlabel("Frame Number", fontsize=22)
    plt.ylabel("Normalized Surprisal Value", fontsize=22)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    save_path = f"~/Desktop/{labels[i]}_surprisals.png"

    plt.savefig(os.path.expanduser(save_path), dpi=300, bbox_inches='tight')
    plt.show() 

    i += 1

if __name__ == "__main__":
    main()
