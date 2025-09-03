import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cpu")


def process_say():
    # import
    results_saycam = torch.load(
        "results/losses_10fs_4_6_8_10_12_14ctxt.pth",
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

    def get_difficulty(scene_index):
        if int(scene_index) < 25:
            return "Easy"
        elif int(scene_index) < 125:
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


def max_accuracy(df, by="Difficulty"):
    group = df.groupby(by)
    accuracies = {}
    for name, subdf in group:
        classifications = list(subdf["classified"].values)
        matrix = torch.stack(classifications).to(torch.float32)
        avgs = torch.mean(matrix, dim=0)
        accuracies[name] = torch.max(avgs)
    return accuracies


def plot_binned_accuracies(saycam_df, meta_df, bin_by):
    saycam_acc = max_accuracy(saycam_df, by=bin_by)
    meta_acc = max_accuracy(meta_df, by=bin_by)

    print("Saycam results:", saycam_acc)
    print("Meta results:", meta_acc)
    labels = sorted(list(set(saycam_acc.keys()) & set(meta_acc.keys())))

    saycam_values = [
        saycam_acc.get(label, torch.tensor(0.0)).item() for label in labels
    ]
    meta_values = [meta_acc.get(label, torch.tensor(0.0)).item() for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, saycam_values, width, label="Saycam")
    rects2 = ax.bar(x + width / 2, meta_values, width, label="Meta")

    ax.set_ylabel("Classification Accuracy")
    ax.set_xlabel(bin_by.capitalize())
    ax.set_title(f"Model Accuracy by {bin_by.capitalize()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)  # Set y-axis from 0 to 1

    # Add bar labels
    ax.bar_label(rects1, padding=3, fmt="%.2f")
    ax.bar_label(rects2, padding=3, fmt="%.2f")

    fig.tight_layout()
    plt.savefig(f"binned_accuracies_by_{bin_by}.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    all_df = pd.DataFrame()

    # process saycam
    saycam_df = process_say()
    saycam_classes = classify(saycam_df)

    # process meta
    meta_df = process_meta()
    meta_classes = classify(meta_df)

    # plots
    plot_binned_accuracies(saycam_classes, meta_classes, bin_by="Difficulty")
    plot_binned_accuracies(saycam_classes, meta_classes, bin_by="condition")
    plot_binned_accuracies(saycam_classes, meta_classes, bin_by="Camera")
    plot_binned_accuracies(saycam_classes, meta_classes, bin_by="SceneIndex")


if __name__ == "__main__":
    main()
