import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cpu")


def process_our_runs(pth, meta_pth):
    # import
    results_saycam = torch.load(
        pth,
        weights_only=False,
        map_location=device,
    )
    meta_saycam = pd.read_csv(meta_pth)

    # index by pair of possible/impossible
    meta_saycam["pair_id"] = meta_saycam["type"].str.split("_").str[0]
    meta_saycam["class"] = meta_saycam["type"].str.split("_").str[1]
    group_cols = ["SceneIndex", "pair_id"]
    meta_saycam["pairs"] = meta_saycam.groupby(group_cols).ngroup()

    # merge losses
    results_df = pd.DataFrame(
        {
            "losses": [x for x in results_saycam["losses"]],
            "name": results_saycam["names"],
        }
    )
    results_df["losses"] = [x[:4] for x in results_df["losses"]]
    results_df["name"] = results_df["name"].str.replace(".mp4", "", regex=False)
    results_df = results_df.drop_duplicates(subset=["name"], keep="first")
    merged_df = pd.merge(meta_saycam, results_df, on="name")

    # add difficulty
    def get_difficulty(scene_index):
        if int(scene_index) <= 25:
            return "Easy"
        elif int(scene_index) <= 125:
            return "Medium"
        else:
            return "Hard"

    merged_df["Difficulty"] = merged_df["SceneIndex"].apply(get_difficulty)

    return merged_df


def process_meta_runs(pth, meta_pth):
    results_0_1 = torch.load(
        pth,
        weights_only=False,
        map_location=device,
    )
    meta_0_1 = pd.read_csv(meta_pth)

    # index by pair
    meta_0_1["trial"] = meta_0_1["filename"].str.split("_").str[:-1].str.join("_")
    meta_0_1["class"] = meta_0_1["type"].str.split("_").str[1]
    meta_0_1["pairs"] = meta_0_1.groupby("trial").ngroup()

    # merge losses
    results_df = pd.DataFrame(
        {"losses": [x for x in results_0_1["losses"]], "filename": results_0_1["names"]}
    )
    results_df = results_df.drop_duplicates(subset=["filename"], keep="first")
    merged_df = pd.merge(meta_0_1, results_df, on="filename")

    # camera
    fixed_map = [
        "FixedJumpSolidity",
        "SolidityFallingFlat",
        "FixedMarryPoppins",
        "RotatingCup",
        "HotAirBallon",
        "SphereFallingDown",
    ]

    merged_df["Camera"] = [
        "Fixed" if x in fixed_map else "Moving" for x in merged_df["game_name"]
    ]

    # merge on condition
    def map_condition(name):
        if "immutability_texture" in name:
            return "immutability"
        elif "continuity_swap" in name:
            return "continuity"
        else:
            return name

    merged_df["condition"] = merged_df["condition"].apply(map_condition)

    # set difficulty
    def get_difficulty(scene_index):
        if int(scene_index) <= 25:
            return "Easy"
        elif int(scene_index) <= 125:
            return "Medium"
        else:
            return "Hard"

    merged_df["Difficulty"] = merged_df["SceneIndex"].apply(get_difficulty)

    return merged_df


def classify(df):
    losses = df["losses"].values
    avg_losses = []
    for v in losses:
        ctx_losses = []
        for context in v:
            ctx_losses.append(torch.mean(context))
        avg_losses.append(ctx_losses)
    df["avg_loss"] = avg_losses

    classifications = []
    for i in range(df["pairs"].max() + 1):
        pair_df = df[df["pairs"] == i]
        imp = torch.tensor(
            pair_df[pair_df["class"] == "Impossible"]["avg_loss"].values[0]
        )
        pos = torch.tensor(
            pair_df[pair_df["class"] == "Possible"]["avg_loss"].values[0]
        )
        classifications.append(imp > pos)

    metadata_by_pair = df.drop_duplicates(subset="pairs", keep="first")
    truncated = metadata_by_pair[
        ["pairs", "SceneIndex", "Camera", "Difficulty", "condition"]
    ]
    merged_df = pd.merge(
        truncated,
        pd.DataFrame(
            {"pairs": range(len(classifications)), "classified": classifications}
        ),
        on="pairs",
    )
    return merged_df


def max_accuracy_with_error(df, by="Difficulty"):
    """
    Groups the dataframe and calculates the max accuracy and the standard error
    of the mean (SEM) for that accuracy.
    """
    group = df.groupby(by)
    accuracies = {}
    for name, subdf in group:
        if len(subdf) == 0:
            continue

        classifications = list(subdf["classified"].values)
        matrix = torch.stack(classifications).to(torch.float32)

        # Calculate mean accuracy for each context window
        avgs = torch.mean(matrix, dim=0)

        # Find the best context window
        best_context_idx = torch.argmax(avgs)
        max_acc = avgs[best_context_idx]

        # Get the classifications for only the best context window
        classifications_for_best_context = matrix[:, best_context_idx]

        # Calculate Standard Error of the Mean (SEM)
        n = len(classifications_for_best_context)
        std_dev = torch.std(classifications_for_best_context)
        sem = std_dev / torch.sqrt(torch.tensor(n, dtype=torch.float32))

        accuracies[name] = (max_acc.item(), sem.item())
    return accuracies


def get_overall_accuracy(df):
    classifications = list(df["classified"].values)
    matrix = torch.stack(classifications).to(torch.float32)
    avgs = torch.mean(matrix, dim=0)
    return torch.max(avgs)


def _plot_single_lollipop_ax_with_errors(
    ax, model_accuracies_with_error, custom_order, title, model_color_map
):
    """Plots a single lollipop chart with error bars on a given matplotlib axis."""
    # X-axis positions for the categories
    x_pos = np.arange(len(custom_order))

    # Calculate offsets for each model so lollipops are side-by-side
    num_models = len(model_accuracies_with_error)
    width = 0.6  # Total width for the group of lollipops
    # Create evenly spaced offsets
    offsets = np.linspace(-width / 2, width / 2, num_models)

    model_names = list(model_accuracies_with_error.keys())

    for i, model_name in enumerate(model_names):
        accs_with_err = model_accuracies_with_error[model_name]

        # Unpack accuracies and errors
        y_values = [accs_with_err.get(category, (0, 0))[0] for category in custom_order]
        y_errors = [accs_with_err.get(category, (0, 0))[1] for category in custom_order]
        # Plot the lollipop "stick"
        offset = offsets[i]
        ax.vlines(
            x_pos + offset,
            ymin=0,
            ymax=y_values,
            color=model_color_map[model_name],
            alpha=0.7,
            linewidth=6,
        )

        # Plot the marker and error bar on top of the stick
        ax.errorbar(
            x=x_pos + offset,
            y=y_values,
            yerr=y_errors,
            fmt="o",  # 'o' is for a circle marker
            color=model_color_map[model_name],
            markersize=8,
            capsize=5,  # Width of the error bar caps
            linestyle="None",  # Do not connect the markers
            label=model_name,
            zorder=3,
        )

        # Add text labels for accuracy values
        for j, y_val in enumerate(y_values):
            # Position text on top of the error bar with a small padding
            ax.text(
                x_pos[j] + offset,
                y_val + y_errors[j] + 0.01,
                f"{y_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="gray",
            )

    # Optional: Plot overall mean accuracy lines (uncomment if needed)
    # for model_name, overall_acc in mean_accuracies.items():
    #     acc_value, _ = overall_acc # Unpack tuple if it contains error
    #     ax.axhline(
    #         y=acc_value,
    #         color=model_color_map[model_name],
    #         linestyle=":",
    #         linewidth=2,
    #         alpha=0.8,
    #     )

    # Plot human performance and chance lines
    ax.axhline(
        y=0.964,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        alpha=1,
        label="Human Performance (0.964)",
    )

    ax.axhline(
        y=0.5,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="Chance (0.5)",
    )
    ax.set_ylim(0.4, 1.05)  # Adjusted ylim to give space for text

    ax.set_xticks(x_pos)
    ax.set_xticklabels([x.capitalize() for x in custom_order], fontsize=20)
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlim(-0.5, len(custom_order) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.8)

    # To avoid duplicate labels in the legend, handle them smartly
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), fontsize=12)


# --- NEW: Combined Plotting Function ---
def plot_combined_charts_with_errors(model_dfs, plot_configs, main_title):
    """
    Generates a single figure with multiple subplots, including error bars.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    width_ratios = [len(config["custom_order"]) for config in plot_configs]

    fig, axs = plt.subplots(
        1,
        len(plot_configs),
        figsize=(7 * len(plot_configs), 5.5),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )

    model_names = list(model_dfs.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
    model_color_map = {name: color for name, color in zip(model_names, colors)}

    for i, config in enumerate(plot_configs):
        ax = axs[i] if len(plot_configs) > 1 else axs
        bin_by = config["bin_by"]
        custom_order = config["custom_order"]

        # --- MODIFIED: Use functions that calculate error ---
        model_accuracies_with_error = {
            name: max_accuracy_with_error(df, by=bin_by)
            for name, df in model_dfs.items()
        }
        # model_overall_accuracies_with_error = {name: get_overall_accuracy_with_error(df) for name, df in model_dfs.items()}
        # --- MODIFIED: Call the plotting function that handles errors ---
        _plot_single_lollipop_ax_with_errors(
            ax,
            model_accuracies_with_error,
            custom_order,
            title=f"by {bin_by.capitalize()}",
            model_color_map=model_color_map,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_linewidth(2)
        ax.spines["left"].set_color("gray")

    fig.suptitle(main_title, fontsize=24, fontweight="bold")
    axs[0].set_ylabel("Accuracy", fontsize=20)

    handles, labels = axs[-1].get_legend_handles_labels()
    # To avoid duplicate labels in the legend, handle them smartly
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.13),
        fontsize=20,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.05, 0.95, 0.96])
    plt.savefig(f"MIN_binned_accuracies_by_condition.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # process saycam
    saycam_df = process_our_runs(
        "results/losses_10fs_4_6_8_10_12_14ctxt.pth", "results/metadata.csv"
    )
    saycam_classes = classify(saycam_df)

    # process meta vjepa
    # meta_df = process_our_runs("results/vjepa-2-h_losses_10fs_4_6_8_10_12_14ctxt.pth", "results/metadata.csv")
    # meta_classes = classify(meta_df)

    meta_df = process_our_runs(
        "results/losses_10fs_12_18_24_30_36_42ctxt.pth", "results/metadata.csv"
    )
    meta_classes = classify(meta_df)

    # process meta vjepa_0_1
    meta_0_1_df = process_meta_runs(
        "0_1/main-10steps_16fpc_-1frames-pred.pth", "0_1/metadata.csv"
    )
    meta_0_1_classes = classify(meta_0_1_df)

    # process mae
    mae_df = process_our_runs(
        "results/mae_say_losses_10fs_4_6_8_10_12_14ctxt.pth", "results/metadata.csv"
    )
    mae_classes = classify(mae_df)

    all_results = {
        "V-JEPA-B-SAYCam (ours)": saycam_classes,
        "V-JEPA-2-H-VM22M": meta_classes,
        "V-JEPA-L-0-1-HowTo100M": meta_0_1_classes,
        "MAE-SAYCam": mae_classes,
    }

    plot_configurations = [
        {
            "bin_by": "condition",
            "custom_order": ["solidity", "immutability", "permanence", "continuity"],
        },
        {"bin_by": "Difficulty", "custom_order": ["Easy", "Medium", "Hard"]},
        {"bin_by": "Camera", "custom_order": ["Fixed", "Moving"]},
    ]

    plot_combined_charts_with_errors(
        all_results, plot_configurations, main_title="Performance by Category"
    )


if __name__ == "__main__":
    main()
