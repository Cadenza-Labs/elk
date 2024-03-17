import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_eval_data(eval_path):
    """
    Load evaluation data from an eval.csv file.
    """
    return pd.read_csv(eval_path)


def prepare_comparison_data(sweep_path1, sweep_path2, datasets):
    """
    Load and prepare comparison data for both sweeps.
    """
    comparison_data = []
    model_creators = set(os.listdir(sweep_path1)) & set(os.listdir(sweep_path2))

    for creator in model_creators:
        models = set(os.listdir(os.path.join(sweep_path1, creator))) & set(
            os.listdir(os.path.join(sweep_path2, creator))
        )
        for model in models:
            for dataset in datasets:
                eval_path1 = os.path.join(
                    sweep_path1, creator, model, dataset, "eval.csv"
                )
                eval_path2 = os.path.join(
                    sweep_path2, creator, model, dataset, "eval.csv"
                )
                df1 = load_eval_data(eval_path1)
                df2 = load_eval_data(eval_path2)
                layers = sorted(set(df1["layer"].unique()) | set(df2["layer"].unique()))
                for layer in layers:
                    # Corrected filtering approach
                    filtered_df1 = df1.loc[
                        (df1["layer"] == layer)
                        & (df1["prompt_ensembling"] == "partial"),
                        "cal_acc_estimate",
                    ]
                    filtered_df2 = df2.loc[
                        (df2["layer"] == layer)
                        & (df2["prompt_ensembling"] == "partial"),
                        "cal_acc_estimate",
                    ]

                    if not filtered_df1.empty and not filtered_df2.empty:
                        cal_acc_diff_sweep1 = filtered_df1.values[0]
                        cal_acc_diff_sweep2 = filtered_df2.values[0]
                        comparison_data.append(
                            {
                                "creator": creator,
                                "model": model,
                                "dataset": dataset,
                                "layer": layer,
                                "cal_acc_diff_sweep1": cal_acc_diff_sweep1,
                                "cal_acc_diff_sweep2": cal_acc_diff_sweep2,
                            }
                        )

    return pd.DataFrame(comparison_data)


def plot_comparison_line_diagram(data, output_dir, sweep_name1, sweep_name2):
    """
    Plot comparison line diagrams for each model across layers for each dataset.
    """
    for model in data["model"].unique():
        model_data = data[data["model"] == model]
        for dataset in model_data["dataset"].unique():
            dataset_data = model_data[model_data["dataset"] == dataset]
            plt.figure(figsize=(10, 6))
            plt.plot(
                dataset_data["layer"],
                dataset_data["cal_acc_diff_sweep1"],
                marker="o",
                label=f"{sweep_name1}",
            )
            plt.plot(
                dataset_data["layer"],
                dataset_data["cal_acc_diff_sweep2"],
                marker="x",
                label=f"{sweep_name2}",
            )
            plt.title(f"Layer-wise Calibrated Accuracy: {model} on {dataset}")
            plt.xlabel("Layer")
            plt.ylabel("Calibrated Accuracy Difference")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{model}_{dataset}.png"))
            plt.close()


def plot_heatmap(data, title, save_path):
    """
    Plot a heatmap of the comparison data and save it to a file.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title, pad=20)
    plt.xlabel("Datasets", labelpad=10)
    plt.ylabel("Models", labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45, va="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_heatmaps(data, output_dir):
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Heatmap for the mean of all layers
    mean_data = data.groupby(["model", "dataset"])["cal_acc_diff"].mean().unstack()
    mean_heatmap_mean_value = (
        mean_data.mean().mean()
    )  # Calculate the global mean of the mean heatmap
    plot_heatmap(
        mean_data,
        "Mean Calibrated Accuracy",
        os.path.join(heatmap_dir, "mean_heatmap.png"),
    )

    # Preparing and plotting heatmaps for the 75th percentile layer
    data["cal_acc_diff"] = data["cal_acc_diff_sweep2"] - data["cal_acc_diff_sweep1"]
    percentile_75_data = (
        data.groupby(["model", "dataset"])["cal_acc_diff"].quantile(0.75).unstack()
    )
    percentile_75_heatmap_mean_value = (
        percentile_75_data.mean().mean()
    )  # Calculate the global mean of the 75th percentile heatmap
    plot_heatmap(
        percentile_75_data,
        "75th Percentile Layer Calibrated Accuracy",
        os.path.join(heatmap_dir, "percentile_75_heatmap.png"),
    )

    # Create a DataFrame to hold the mean values
    heatmap_means = pd.DataFrame(
        {
            "Heatmap Type": [
                "Mean Calibrated Accuracy",
                "75th Percentile Calibrated Accuracy",
            ],
            "Global Mean Value": [
                mean_heatmap_mean_value,
                percentile_75_heatmap_mean_value,
            ],
        }
    )

    # Save the DataFrame as a CSV file
    heatmap_means.to_csv(
        os.path.join(heatmap_dir, "heatmap_global_means.csv"), index=False
    )


def main(sweep_path1, sweep_path2, output_dir):
    datasets = ["imdb", "amazon_polarity"]  # Specify your datasets
    # The output_dir variable is now passed directly to the function
    # and used in the generation of heatmaps and comparison diagrams.

    sweep_name1 = os.path.basename(os.path.normpath(sweep_path1))
    sweep_name2 = os.path.basename(os.path.normpath(sweep_path2))

    comparison_data = prepare_comparison_data(sweep_path1, sweep_path2, datasets)
    comparison_data["cal_acc_diff"] = (
        comparison_data["cal_acc_diff_sweep2"] - comparison_data["cal_acc_diff_sweep1"]
    )
    generate_heatmaps(comparison_data, output_dir)
    plot_comparison_line_diagram(comparison_data, output_dir, sweep_name1, sweep_name2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Models Across Sweeps")
    parser.add_argument("sweep_path1", type=str, help="Path to the first sweep folder")
    parser.add_argument("sweep_path2", type=str, help="Path to the second sweep folder")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_comparisons",
        help="Root folder for saving the visualizations",
    )
    args = parser.parse_args()
    main(args.sweep_path1, args.sweep_path2, args.output_dir)
