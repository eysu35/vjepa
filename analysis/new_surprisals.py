import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random


def map_name_to_hash(name, say_meta):
    try:
        attributes = name.split("_")
        cond1 = say_meta["SceneIndex"] == int(attributes[0])
        cond2 = say_meta["pairIdx"] == attributes[7]
        cond3 = say_meta["type2"] == attributes[8]
        matching_hash = say_meta[cond1][cond2][cond3]
        return matching_hash["name"].item()
    except:
        return None


def all_correct(say, small, meta):
    successful_say = say[say["success"] == 1].copy()
    successful_meta = meta[meta["success"] == 1].copy()
    successful_small = small[small["success"] == 1].copy()
    successful_small = successful_small.dropna(subset=["hash_p"])

    merged_say_meta = pd.merge(
        successful_say, successful_meta, on="name_p", suffixes=("_say", "_meta")
    )
    all_successful = pd.merge(
        merged_say_meta,
        successful_small,
        left_on="name_p",  # Key from the left DataFrame (merged_say_meta)
        right_on="hash_p",  # Key from the right DataFrame (successful_small)
    )

    print(
        f"Found {len(all_successful)} hashes that were successful in all three DataFrames."
    )
    return all_successful


def all_incorrect(say, small, meta):
    successful_say = say[say["success"] == 0].copy()
    successful_meta = meta[meta["success"] == 0].copy()
    successful_small = small[small["success"] == 0].copy()
    successful_small = successful_small.dropna(subset=["hash_p"])

    merged_say_meta = pd.merge(
        successful_say, successful_meta, on="name_p", suffixes=("_say", "_meta")
    )
    all_successful = pd.merge(
        merged_say_meta,
        successful_small,
        left_on="name_p",  # Key from the left DataFrame (merged_say_meta)
        right_on="hash_p",  # Key from the right DataFrame (successful_small)
    )

    print(
        f"Found {len(all_successful)} hashes that were unsuccessful in all three DataFrames."
    )
    return all_successful  # this is act all unsuccessful


def _plot_single_model_panel(
    ax: plt.Axes, data_row: pd.Series, model_info: Dict, classification_status: str
):
    model_name = model_info["name"]
    suffix = model_info["suffix"]

    losses_p_col = f"losses_p{suffix}"
    losses_imp_col = f"losses_imp{suffix}"

    losses_p = data_row[losses_p_col]
    losses_imp = data_row[losses_imp_col]

    last_frame_p = (losses_p > 0).nonzero(as_tuple=True)[0].max().item() + 1
    last_frame_imp = (losses_imp > 0).nonzero(as_tuple=True)[0].max().item() + 1
    losses_p_trimmed = losses_p[:last_frame_p]
    losses_imp_trimmed = losses_imp[:last_frame_imp]

    ax.plot(losses_p_trimmed.numpy(), label="Possible", color="royalblue", alpha=0.8)
    ax.plot(
        losses_imp_trimmed.numpy(), label="Impossible", color="darkorange", alpha=0.8
    )

    ax.axhline(
        y=torch.mean(losses_p_trimmed), color="royalblue", linestyle="--", linewidth=1.5
    )
    ax.axhline(
        y=torch.mean(losses_imp_trimmed),
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_title(f"{model_name}\n({classification_status})", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_comparison_from_dfs(
    correct_df: pd.DataFrame,
    incorrect_df: pd.DataFrame,
    output_filename="model_comparison_from_df.png",
):
    correct_example_row = correct_df.sample(n=1).iloc[0]
    incorrect_example_row = incorrect_df.sample(n=1).iloc[0]

    print(correct_example_row)
    print(incorrect_example_row)

    models = [
        {"name": "V-JEPA-B-SAYCam (ours)", "suffix": "_say"},
        {"name": "V-JEPA-H-VM22M", "suffix": "_meta"},
        {"name": "V-JEPA-L-0-1-HowTo100M", "suffix": ""},
    ]

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(16, 8), sharey=False, squeeze=False
    )
    fig.suptitle("Frame-by-Frame Surprise", fontsize=24, y=1.0)

    for i, model_info in enumerate(models):
        ax_top = axes[0, i]
        _plot_single_model_panel(ax_top, correct_example_row, model_info, "Correct")
        ax_bottom = axes[1, i]
        _plot_single_model_panel(
            ax_bottom, incorrect_example_row, model_info, "Incorrect"
        )
        ax_bottom.set_xlabel("Frame", fontsize=20)
        if i == 0:
            ax_top.set_ylabel("Surprise", fontsize=20)
            ax_bottom.set_ylabel("Surprise", fontsize=20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", fontsize=20, bbox_to_anchor=(1, 1.05)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("frame_by_frame_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # our model
    say_file_path = "surprise_results/vjepa_b_losses_10fs_4_6_8_10_12_14ctxt_best_predictions_avg.pth"
    say_preds = torch.load(say_file_path)
    say_df = pd.DataFrame(say_preds)

    say_meta = pd.read_csv("results/metadata.csv")
    say_meta["SceneIndex"] = say_meta["SceneIndex"].apply(int)
    say_meta["pairIdx"] = [x.split("_")[0] for x in say_meta["type"]]
    say_meta["type2"] = [x.split("_")[1] for x in say_meta["type"]]

    # small meta model
    small_meta_file_path = (
        "surprise_results/vjepa_l_losses_10fs_4_6_8_10ctxt_best_predictions_avg.pth"
    )
    small_preds = torch.load(small_meta_file_path)
    small_df = pd.DataFrame(small_preds)
    small_df["hash_p"] = [
        map_name_to_hash(x, say_meta) for x in small_df["name_p"]
    ]  # map to hash
    small_meta = pd.read_csv("0_1/metadata.csv")

    # big meta model
    meta_file_path = "surprise_results/vjepa_h_losses_10fs_4_6_8_10_12_14ctxt_best_predictions_avg.pth"
    meta_preds = torch.load(meta_file_path)
    meta_df = pd.DataFrame(meta_preds)

    correct_df = all_correct(say_df, small_df, meta_df)
    incorrect_df = all_incorrect(say_df, small_df, meta_df)

    plot_comparison_from_dfs(correct_df, incorrect_df)
