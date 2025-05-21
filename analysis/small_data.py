
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pathlib import Path
root = Path("./data_intphys")

categories = [
        "Object Permanence",
        "Continuity",
        "Shape Constancy",
        "Color Constancy",
        "Support",
        "Inertia",
        "Gravity",
        "Solidity",
        "Collision",
]
mappings = {
    "intphys":{'O1':["Object Permanence"],
               'O2':["Shape Constancy"],
               'O3':["Continuity"]},
    "inflevel":{'continuity':["Object Permanence"],
                'solidity':["Solidity"],
                'gravity':["Gravity"]},
    "grasp":{'Collision':['Collision'],
            'Continuity':['Continuity'],
            'Gravity':['Gravity'],
            'GravityContinuity':['Gravity','Continuity'],
            'GravityInertia':['Gravity','Inertia'],
            'GravityInertia2':['Gravity','Inertia'],
            'GravitySupport':['Gravity','Support'],
            'Inertia':['Inertia'],
            'Inertia2':['Inertia'],
            'ObjectPermanence':['Object Permanence'],
            'ObjectPermanence2':['Object Permanence'],
            'ObjectPermanence3':['Object Permanence'],
            'SolidityContinuity':['Solidity','Continuity'],
            'SolidityContinuity2':['Solidity','Continuity'],
            'Unchangeableness':['Color Constancy'],
            'Unchangeableness2':['Color Constancy']}}

DEFAULT_CATEGORY_COLORS = ["#3292a8","#ffa600","#994636","#a6d3a0","#d4b483","#7d1128","#b4cded","#b279a7","#bfcc94"] * 10


def get_perf(root,exp,dataset,key_metric="Relative Accuracy (avg)",categories=["Object Permanence","Continuity","Shape Constancy","Color Constancy","Support","Gravity","Solidity","Inertia","Collision",]):
    perf_by_category_model = { prop:[] for prop in categories}
    path = root/exp/dataset/"performance.csv"
    df = pd.read_csv(path,sep=';',skiprows=0)
    props = df["Block"].unique()
    vals = []
    for prop in props:
        targets = mappings[dataset][prop]
        acc=df[df["Block"] == prop].max()[key_metric]
        for key in targets:
            perf_by_category_model[key].append(acc)
    return perf_by_category_model

def plot_model_comparison_scatter_concise(
    model1_accs,
    model2_accs,
    model1_name="Model 1",
    model2_name="Model 2",
    title="Accuracy",
    dataset = "intphys",
    save_fig= "model_comparison.png",
    performance_metric_name="Accuracy",
    category_colors=None,
    annotate_points=True,
    add_diagonal_line=True,
    figsize=(8,8), # Slightly smaller default figure size
    marker_size=80,
    font_size_title=20,
    font_size_labels=16,
    font_size_annotations=16
):
    """Minimal scatter plot comparing two models by category."""
    common_categories = sorted(list(set(model1_accs.keys()) & set(model2_accs.keys())))

    plt.figure(figsize=figsize)

    if not common_categories:
        print("Warning: No common categories found.")
        plt.title(title, fontsize=font_size_title)
        plt.xlabel(f"{model1_name} - {performance_metric_name} (%)", fontsize=font_size_labels)
        plt.ylabel(f"{model2_name} - {performance_metric_name} (%)", fontsize=font_size_labels)
        plt.grid(True, linestyle="--", alpha=0.5)
        if add_diagonal_line:
            plt.plot([0, 100], [0, 100], 'k--', lw=1, label="Equal Perf.")
            plt.legend(loc='lower right')
        plt.xlim([0, 100]); plt.ylim([0, 100])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        return

    m1_vals = [model1_accs[cat] for cat in common_categories]
    m2_vals = [model2_accs[cat] for cat in common_categories]

    # Determine colors for points
    active_colors_config = category_colors if category_colors is not None else DEFAULT_CATEGORY_COLORS
    point_colors = []
    if isinstance(active_colors_config, dict):
        default_color_for_missing = DEFAULT_CATEGORY_COLORS[0]
        point_colors = [active_colors_config.get(cat, default_color_for_missing) for cat in common_categories]
    elif isinstance(active_colors_config, list) and active_colors_config: # Non-empty list
        point_colors = [active_colors_config[i % len(active_colors_config)] for i in range(len(common_categories))]
    else: # Single color string or fallback for empty list/other types
        default_color = str(active_colors_config) if isinstance(active_colors_config, str) else DEFAULT_CATEGORY_COLORS[0]
        point_colors = [default_color] * len(common_categories)

    plt.scatter(m1_vals, m2_vals, c=point_colors, s=marker_size, alpha=0.75, edgecolors='black', linewidth=0.5, zorder=3)

    if annotate_points:
        for i, category_name in enumerate(common_categories):
            plt.annotate(category_name, (m1_vals[i], m2_vals[i]), textcoords="offset points", xytext=(4,0), ha='left', va='center', fontsize=font_size_annotations)

    # Determine plot limits (assuming metric is often % based, 0-100)
    all_values = m1_vals + m2_vals
    data_min, data_max = min(all_values), max(all_values)
    value_range = data_max - data_min
    padding = value_range * 0.1 if value_range > 0 else 2.0 # 10% padding or 2 units if no range

    plot_lim_min = max(0, data_min - padding)
    plot_lim_max = min(100, data_max + padding)
    if plot_lim_max <= plot_lim_min + 1: # Ensure some visible range if data is clustered
        plot_lim_min = max(0, data_min - 2)
        plot_lim_max = min(100, data_max + 2 if data_max > data_min else data_min + 2)

    if add_diagonal_line:
        plt.plot([plot_lim_min, plot_lim_max], [plot_lim_min, plot_lim_max], 'k--', lw=1.2, alpha=0.7, label="Equal Perf.", zorder=2)
        legend = plt.legend(loc='lower right', frameon=True, framealpha=0.85, facecolor='white', edgecolor='gray')

    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.title(dataset, fontsize=font_size_title)
    plt.xlabel(f"{model1_name}", fontsize=font_size_labels)
    plt.ylabel(f"{model2_name}", fontsize=font_size_labels)

    plt.xlim([plot_lim_min, plot_lim_max])
    plt.ylim([plot_lim_min, plot_lim_max])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    print("saving figure...")
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    plt.show()

# Example Usage:
def main():
  key_metric = "Relative Accuracy (avg)"

  print("Loading data...")
  for dataset in ["inflevel", "grasp", "intphys"]:
    perfs_vid = []
    perf_by_category = { prop:[] for prop in categories}
    for exp in ["vit-l-rope-howto-0_1p_vid", "vjepa_say_base_no_rope"]:
        perf = {}
        perf_by_category_model = get_perf(root,exp,dataset,key_metric=key_metric,categories=categories)
        for key,value in perf_by_category_model.items():
            if value != []:
                perf[key] = np.mean(value)
        perfs_vid.append(perf)

    print(perfs_vid)
    modelA_perf_data = perfs_vid[0]
    modelB_perf_data = perfs_vid[1]

    print("plotting data...")
    plot_model_comparison_scatter_concise(
        model1_accs=modelA_perf_data,
        model2_accs=modelB_perf_data,
        model1_name="V-JEPA (VideoMix2M)",
        model2_name="V-JEPA (SAYCam)",
        title="Pairwise Classification Accuracy",
        dataset=dataset,
        save_fig=f"compare_{dataset}.png",
        category_colors=["#e6194B", "#3cb44b", "#ffe119"],
        performance_metric_name="",
        annotate_points=True
    )

if __name__ == '__main__':
    main()
