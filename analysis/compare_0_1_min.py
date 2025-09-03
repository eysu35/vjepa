import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cpu")


def process_say(pth):
    # import
    results_saycam = torch.load(
        pth,
        weights_only=False,
        map_location=device,
    )
    meta_saycam = pd.read_csv("results/metadata.csv")

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


def process_meta():
    results_0_1 = torch.load(
        "0_1/main-10steps_16fpc_-1frames-pred.pth",
        weights_only=False,
        map_location=device,
    )
    meta_0_1 = pd.read_csv("0_1/metadata.csv")

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
        min_loss = torch.min(v, dim=0)
        avg_losses.append(torch.mean(min_loss.values).item())
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


def max_accuracy(df, by="Difficulty"):
    group = df.groupby(by)
    accuracies = {}
    for name, subdf in group:
        classifications = list(subdf["classified"].values)
        matrix = torch.stack(classifications).to(torch.float32)
        avgs = torch.mean(matrix, dim=0)
        accuracies[name] = torch.max(avgs)
    return accuracies


def plot_lollipop_chart(saycam_df, meta_df, bin_by, custom_order=None, title=""):
    """
    Generates a professional-looking lollipop chart to compare binned accuracies.
    """
    # 1. Calculate accuracies
    saycam_acc = max_accuracy(saycam_df, by=bin_by)
    meta_acc = max_accuracy(meta_df, by=bin_by)

    # 2. Prepare data for plotting using the custom order
    labels = (
        custom_order
        if custom_order
        else sorted(list(set(saycam_acc.keys()) & set(meta_acc.keys())))
    )
    saycam_values = [
        saycam_acc.get(label, torch.tensor(0.0)).item() for label in labels
    ]
    meta_values = [meta_acc.get(label, torch.tensor(0.0)).item() for label in labels]

    # 3. Set up plot aesthetics
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a clean style
    fig, ax = plt.subplots(figsize=(8, 5))  # A more compact figure size

    # Define positions and a slight offset for the two datasets
    x = np.arange(len(labels))
    offset = 0.15

    # Professional color palette
    saycam_color = "#348ABD"  # A nice blue
    meta_color = "#A60628"  # A complementary red/maroon

    # 4. Plot the data as lollipop charts
    # Vertical lines (the "sticks")
    ax.vlines(
        x - offset,
        ymin=0,
        ymax=saycam_values,
        color=saycam_color,
        alpha=0.7,
        linewidth=2,
    )
    ax.vlines(
        x + offset, ymin=0, ymax=meta_values, color=meta_color, alpha=0.7, linewidth=2
    )

    # Dots (the "candy")
    ax.scatter(
        x - offset, saycam_values, color=saycam_color, s=120, label="Saycam", zorder=3
    )
    ax.scatter(x + offset, meta_values, color=meta_color, s=120, label="Meta", zorder=3)

    # 5. Add data labels to the dots
    for i in range(len(labels)):
        saycam_val = saycam_values[i]
        meta_val = meta_values[i]
        ax.text(
            i - offset,
            saycam_val + 0.02,
            f"{saycam_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=saycam_color,
        )
        ax.text(
            i + offset,
            meta_val + 0.02,
            f"{meta_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=meta_color,
        )

    # 6. Refine the plot's appearance
    # Set labels and title with adjusted font sizes
    ax.set_ylabel("Max Context Accuracy", fontsize=12)
    ax.set_title(f"Accuracy By {bin_by}", fontsize=14)

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # Set y-axis limits and add a horizontal line at y=0.5 for reference
    ax.set_ylim(0, 1.05)
    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Chance (0.5)",
    )

    # Improve the legend
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="gray")

    # Remove unnecessary spines (borders) for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(
        f"MIN_lollipop_accuracies_by_{bin_by}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    all_df = pd.DataFrame()

    # process saycam
    saycam_df = process_say("results/losses_10fs_4_6_8_10_12_14ctxt.pth")
    saycam_classes = classify(saycam_df)

    # process meta vjepa
    meta_df = process_say("results/losses_10fs_12_18_24_30_36_42ctxt.pth")
    meta_classes = classify(meta_df)

    # process meta vjepa_0_1
    # meta_df = process_meta()
    # meta_classes = classify(meta_df)

    # plots
    order = ["Easy", "Medium", "Hard"]
    plot_lollipop_chart(
        saycam_classes, meta_classes, bin_by="Difficulty", custom_order=order
    )

    order = ["solidity", "immutability", "permanence", "continuity"]
    plot_lollipop_chart(
        saycam_classes, meta_classes, bin_by="condition", custom_order=order
    )

    order = ["Fixed", "Moving"]
    plot_lollipop_chart(
        saycam_classes, meta_classes, bin_by="Camera", custom_order=order
    )


if __name__ == "__main__":
    main()
