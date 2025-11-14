import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from scipy.stats import spearmanr


def load_all_trajectories(log_dir: Path) -> Tuple[Dict[float, List[Dict]], Dict[str, str]]:
    trajectories = {}
    model_info = {}

    trajectory_files = list(log_dir.rglob("trajectory.json"))
    print(f"found {len(trajectory_files)} files")

    for traj_file in trajectory_files:
        with open(traj_file, "r") as f:
            data = json.load(f)

        alpha = data["alpha"]
        trajectory = data["trajectory"]

        if not model_info:
            model_info["encoder"] = data.get("model", "unknown")
            model_info["decoder"] = data.get("decoder", "unknown")

        if alpha not in trajectories:
            trajectories[alpha] = []

        trajectories[alpha].extend(trajectory)

    trajectories = dict(sorted(trajectories.items()))

    print(f"loaded {len(trajectories)} alphas")
    for alpha, traj in trajectories.items():
        print(f"  α={alpha:>8.5f}: {len(traj)} pts")

    return trajectories, model_info


def compute_reconstruction_gap(trajectories: Dict[float, List[Dict]], n_bins: int = 15) -> Tuple[float, List[Dict]]:
    all_points = []
    for alpha, traj in trajectories.items():
        for point in traj:
            all_points.append((point["mse"], point["accuracy"]))

    if len(all_points) == 0:
        print("no data")
        return 0.0, []

    all_mse = np.array([mse for mse, _ in all_points])

    bin_edges = np.percentile(all_mse, np.linspace(0, 100, n_bins + 1))

    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        print("not enough mse values")
        return 0.0, []

    gap = 0.0
    bin_stats = []

    for i in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]

        bin_accs = [acc for mse, acc in all_points if bin_min <= mse < bin_max]

        if len(bin_accs) >= 2:
            spread = max(bin_accs) - min(bin_accs)
            gap += spread

            bin_stats.append({
                "bin_index": i,
                "mse_min": float(bin_min),
                "mse_max": float(bin_max),
                "n_points": len(bin_accs),
                "acc_min": float(min(bin_accs)),
                "acc_max": float(max(bin_accs)),
                "acc_spread": float(spread),
                "acc_mean": float(np.mean(bin_accs)),
                "acc_std": float(np.std(bin_accs)),
            })

    return gap, bin_stats


def plot_tube_scatter(trajectories: Dict[float, List[Dict]], output_path: Path, model_info: Dict[str, str] = None, title: str = "MAE Reconstruction Tube"):
    all_mse = []
    all_acc = []
    all_alpha_labels = []
    all_epochs = []
    all_lrs = []

    for alpha, traj in trajectories.items():
        for point in traj:
            all_mse.append(point["mse"])
            all_acc.append(point["accuracy"])
            all_alpha_labels.append(alpha)
            all_epochs.append(point["epoch"])
            all_lrs.append(point.get("lr", 1.5e-4))

    all_mse = np.array(all_mse)
    all_acc = np.array(all_acc)
    all_alpha_labels = np.array(all_alpha_labels)
    all_epochs = np.array(all_epochs)
    all_lrs = np.array(all_lrs)

    fig, ax = plt.subplots(figsize=(14, 8))

    unique_epoch_lr_pairs = set((e, lr) for e, lr in zip(all_epochs, all_lrs))
    for epoch, lr in sorted(unique_epoch_lr_pairs):
        mask = (all_epochs == epoch) & (all_lrs == lr)
        epoch_mses = all_mse[mask]
        epoch_accs = all_acc[mask]
        epoch_alphas = all_alpha_labels[mask]

        sort_idx = np.argsort(epoch_alphas)
        sorted_mses = epoch_mses[sort_idx]
        sorted_accs = epoch_accs[sort_idx]

        ax.plot(sorted_mses, sorted_accs, 'k--', alpha=0.15, linewidth=0.5, zorder=1)

    from matplotlib import cm
    cmap = plt.colormaps['plasma']
    norm = plt.matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)

    colors = []
    for alpha_val in all_alpha_labels:
        if abs(alpha_val) < 1e-9:
            colors.append('white')
        else:
            colors.append(cmap(norm(alpha_val)))

    scatter = ax.scatter(
        all_mse, all_acc,
        c=colors,
        s=30,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        zorder=2
    )

    corr, p_value = spearmanr(all_mse, all_acc)

    n_bins = 15
    mse_bins = np.linspace(all_mse.min(), all_mse.max(), n_bins+1)
    tube_volume = 0.0
    for i in range(len(mse_bins)-1):
        mask = (all_mse >= mse_bins[i]) & (all_mse < mse_bins[i+1])
        if mask.sum() > 3:
            acc_in_bin = all_acc[mask]
            tube_volume += acc_in_bin.max() - acc_in_bin.min()

    unique_alphas = np.unique(all_alpha_labels)
    unique_epochs = np.unique(all_epochs)
    unique_lrs = np.unique(all_lrs)

    # Parse encoder/decoder specs from new naming format (e.g., enc320d8h5p16, dec192d4h3)
    def parse_model_spec(name):
        import re
        # Match patterns like enc320d8h5p16 or dec192d4h3
        match = re.match(r'(enc|dec)(\d+)d(\d+)h(\d+)(?:p(\d+))?', name)
        if match:
            _, dim, depth, heads, patch = match.groups()
            spec = f'd={dim}, L={depth}, h={heads}'
            if patch:
                spec += f', p={patch}'
            return spec
        # Fallback to old hardcoded specs
        old_encoder_specs = {"vit_tiny": "d=192, L=12, h=3", "vit_small": "d=384, L=12, h=6",
                           "vit_base": "d=768, L=12, h=12", "vit_large": "d=1024, L=24, h=16"}
        old_decoder_specs = {"base-8b": "d=512, L=8, h=16", "tiny-4b": "d=192, L=4, h=3",
                           "linear": "linear decoder"}
        return old_encoder_specs.get(name) or old_decoder_specs.get(name) or ""

    textstr = ''
    if model_info:
        enc = model_info.get("encoder", "unknown")
        dec = model_info.get("decoder", "unknown")
        enc_spec = parse_model_spec(enc)
        dec_spec = parse_model_spec(dec)
        textstr += f'Encoder: {enc}\n'
        if enc_spec:
            textstr += f'  {enc_spec}\n'
        textstr += f'Decoder: {dec}\n'
        if dec_spec:
            textstr += f'  {dec_spec}\n'
        textstr += '---\n'

    textstr += f'Data: {len(all_mse)} validation points\n'
    textstr += f'LRs: {len(unique_lrs)}, Lambdas: {len(unique_alphas)}, Epochs: {len(unique_epochs)}\n'
    textstr += f'Spearman ρ: {corr:.3f} (p={p_value:.2e})\n'
    textstr += f'Tube Volume Γ: {tube_volume:.1f}%'

    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.9,
                 edgecolor='black', linewidth=2)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

    ax.set_xlabel('Reconstruction MSE (Validation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Lambda (Supervision Strength)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"saved: {output_path}")


def plot_alpha_vs_metrics(trajectories: Dict[float, List[Dict]], output_path: Path, model_info: Dict[str, str] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    alphas = []
    final_mses = []
    final_accs = []

    for alpha in sorted(trajectories.keys()):
        traj = trajectories[alpha]
        if len(traj) > 0:
            final = traj[-1]
            alphas.append(alpha)
            final_mses.append(final["mse"])
            final_accs.append(final["accuracy"])

    ax1.plot(alphas, final_mses, 'o-', markersize=8, linewidth=2, color='blue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='λ=0 (Pure MAE)')
    ax1.set_xlabel("Lambda", fontsize=12)
    ax1.set_ylabel("Final Val Reconstruction MSE", fontsize=12)
    ax1.set_title("Reconstruction MSE vs Lambda", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(alphas, final_accs, 'o-', markersize=8, linewidth=2, color='green')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='λ=0 (Pure MAE)')
    ax2.set_xlabel("Lambda", fontsize=12)
    ax2.set_ylabel("Final Val Linear Probe Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Lambda", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    title = "Lambda Negative to Positive 'Phase Transition'"
    if model_info:
        encoder_specs = {"vit_tiny": "d=192, L=12, h=3", "vit_small": "d=384, L=12, h=6",
                         "vit_base": "d=768, L=12, h=12", "vit_large": "d=1024, L=24, h=16"}
        decoder_specs = {"base-8b": "d=512, L=8, h=16", "tiny-4b": "d=192, L=4, h=3"}
        enc = model_info.get("encoder", "unknown")
        dec = model_info.get("decoder", "unknown")
        enc_spec = encoder_specs.get(enc, "")
        dec_spec = decoder_specs.get(dec, "")
        subtitle = f"Encoder: {enc} ({enc_spec}) | Decoder: {dec} ({dec_spec})"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        fig.text(0.5, 0.925, subtitle, ha='center', fontsize=10, style='italic')
    else:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved: {output_path}")


def plot_negative_vs_positive_comparison(trajectories: Dict[float, List[Dict]], output_path: Path, model_info: Dict[str, str] = None):
    fig, ax = plt.subplots(figsize=(12, 8))

    all_mse = []
    all_acc = []
    all_alpha_labels = []
    all_epochs = []
    all_lrs = []

    for alpha, traj in trajectories.items():
        for point in traj:
            all_mse.append(point["mse"])
            all_acc.append(point["accuracy"])
            all_alpha_labels.append(alpha)
            all_epochs.append(point["epoch"])
            all_lrs.append(point.get("lr", 1.5e-4))

    all_mse = np.array(all_mse)
    all_acc = np.array(all_acc)
    all_alpha_labels = np.array(all_alpha_labels)
    all_epochs = np.array(all_epochs)
    all_lrs = np.array(all_lrs)

    unique_epoch_lr_pairs = set((e, lr) for e, lr in zip(all_epochs, all_lrs))
    for epoch, lr in sorted(unique_epoch_lr_pairs):
        mask = (all_epochs == epoch) & (all_lrs == lr)
        epoch_mses = all_mse[mask]
        epoch_accs = all_acc[mask]
        epoch_alphas = all_alpha_labels[mask]

        sort_idx = np.argsort(epoch_alphas)
        sorted_mses = epoch_mses[sort_idx]
        sorted_accs = epoch_accs[sort_idx]

        ax.plot(sorted_mses, sorted_accs, 'k--', alpha=0.15, linewidth=0.5, zorder=1)

    negative_alphas = {a: t for a, t in trajectories.items() if a < 0}
    zero_alpha = {a: t for a, t in trajectories.items() if a == 0}
    positive_alphas = {a: t for a, t in trajectories.items() if a > 0}

    for alpha, traj in negative_alphas.items():
        mses = [p["mse"] for p in traj]
        accs = [p["accuracy"] for p in traj]
        ax.scatter(mses, accs, c='red', alpha=0.5, s=30, label="λ<0" if alpha in [min(negative_alphas.keys())] else "", zorder=2)

    for alpha, traj in zero_alpha.items():
        mses = [p["mse"] for p in traj]
        accs = [p["accuracy"] for p in traj]
        ax.scatter(mses, accs, c='black', alpha=0.8, s=50, marker='x', label="λ=0 (Pure MAE)", zorder=2)

    for alpha, traj in positive_alphas.items():
        mses = [p["mse"] for p in traj]
        accs = [p["accuracy"] for p in traj]
        ax.scatter(mses, accs, c='blue', alpha=0.5, s=30, label="λ>0" if alpha in [max(positive_alphas.keys())] else "", zorder=2)

    ax.set_xlabel("Reconstruction MSE", fontsize=14)
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=14)

    title = "Negative vs Positive Lambda Comparison"
    if model_info:
        encoder_specs = {"vit_tiny": "d=192, L=12, h=3", "vit_small": "d=384, L=12, h=6",
                         "vit_base": "d=768, L=12, h=12", "vit_large": "d=1024, L=24, h=16"}
        decoder_specs = {"base-8b": "d=512, L=8, h=16", "tiny-4b": "d=192, L=4, h=3"}
        enc = model_info.get("encoder", "unknown")
        dec = model_info.get("decoder", "unknown")
        enc_spec = encoder_specs.get(enc, "")
        dec_spec = decoder_specs.get(dec, "")
        subtitle = f"Encoder: {enc} ({enc_spec}) | Decoder: {dec} ({dec_spec})"
        title = f"{title}\n{subtitle}"

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved: {output_path}")


def save_gap_statistics(trajectories: Dict[float, List[Dict]], gap: float, bin_stats: List[Dict], output_path: Path):
    alpha_correlations = {}
    for alpha, traj in trajectories.items():
        if len(traj) >= 3:
            mses = [p["mse"] for p in traj]
            accs = [p["accuracy"] for p in traj]
            corr, pval = spearmanr(mses, accs)
            alpha_correlations[float(alpha)] = {
                "spearman_r": float(corr),
                "p_value": float(pval),
            }

    all_mses = [p["mse"] for traj in trajectories.values() for p in traj]
    all_accs = [p["accuracy"] for traj in trajectories.values() for p in traj]

    overall_corr, overall_pval = spearmanr(all_mses, all_accs) if len(all_mses) >= 3 else (0, 1)

    stats = {
        "reconstruction_gap": float(gap),
        "n_bins": len(bin_stats),
        "bin_statistics": bin_stats,
        "alpha_correlations": alpha_correlations,
        "overall_correlation": {
            "spearman_r": float(overall_corr),
            "p_value": float(overall_pval),
        },
        "summary": {
            "n_experiments": len(trajectories),
            "total_points": len(all_mses),
            "mse_range": [float(min(all_mses)), float(max(all_mses))],
            "acc_range": [float(min(all_accs)), float(max(all_accs))],
        }
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"saved: {output_path}")
    print(f"\n{'='*60}")
    print("gap analysis")
    print(f"{'='*60}")
    print(f"gap: {gap:.2f}%")
    print(f"corr: {overall_corr:.4f} (p={overall_pval:.4e})")
    print(f"bins: {len(bin_stats)}")
    print(f"exps: {len(trajectories)}")
    print(f"pts: {len(all_mses)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze MAE reconstruction gap experiments")
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=["imagenette", "tiny-imagenet", "imagenet100", "imagenet"],
                        help="Dataset name")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory containing experiment logs (default: outputs/{dataset}-mae-reconstruction-gap)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: outputs/{dataset}-analysis)")
    parser.add_argument("--n_bins", type=int, default=15,
                        help="Number of MSE bins for gap computation")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (default: {dataset}-mae-reconstruction-gap)")

    args = parser.parse_args()

    if args.log_dir is None:
        args.log_dir = f"outputs/{args.dataset}-mae-reconstruction-gap"
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.dataset}-analysis"
    if args.wandb_project is None:
        args.wandb_project = f"{args.dataset}-mae-reconstruction-gap"

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading from {log_dir}")

    trajectories, model_info = load_all_trajectories(log_dir)

    if len(trajectories) == 0:
        print("no trajectories")
        return

    print("\ncomputing gap...")
    gap, bin_stats = compute_reconstruction_gap(trajectories, n_bins=args.n_bins)

    print("\ngenerating plots...")
    tube_plot_path = output_dir / "tube_scatter_signed_alpha.png"
    alpha_metrics_path = output_dir / "alpha_vs_final_metrics.png"
    comparison_path = output_dir / "negative_vs_positive_comparison.png"
    stats_path = output_dir / "gap_statistics.json"

    plot_tube_scatter(trajectories, tube_plot_path, model_info)
    plot_alpha_vs_metrics(trajectories, alpha_metrics_path, model_info)
    plot_negative_vs_positive_comparison(trajectories, comparison_path, model_info)

    print("\nsaving stats...")
    save_gap_statistics(trajectories, gap, bin_stats, stats_path)

    print(f"\ndone, saved to {output_dir}")

    if args.wandb_project:
        print(f"\nuploading to wandb: {args.wandb_project}")
        try:
            import wandb

            all_mses = [p["mse"] for traj in trajectories.values() for p in traj]
            all_accs = [p["accuracy"] for traj in trajectories.values() for p in traj]
            overall_corr, overall_pval = spearmanr(all_mses, all_accs) if len(all_mses) >= 3 else (0, 1)

            wandb.init(
                project=args.wandb_project,
                name="reconstruction_gap_analysis",
                job_type="analysis",
                config={
                    "n_bins": args.n_bins,
                    "n_experiments": len(trajectories),
                    "total_points": len(all_mses),
                    "alphas": sorted(trajectories.keys()),
                }
            )

            wandb.log({
                "analysis/tube_scatter": wandb.Image(str(tube_plot_path)),
                "analysis/alpha_vs_metrics": wandb.Image(str(alpha_metrics_path)),
                "analysis/negative_vs_positive_comparison": wandb.Image(str(comparison_path)),
            })

            artifact = wandb.Artifact(
                name="reconstruction_gap_statistics",
                type="analysis",
                description="Reconstruction gap statistics including bin details and correlations"
            )
            artifact.add_file(str(stats_path))
            wandb.log_artifact(artifact)

            wandb.summary.update({
                "tube_volume_gamma": gap,
                "spearman_correlation": overall_corr,
                "spearman_p_value": overall_pval,
                "n_experiments": len(trajectories),
                "total_data_points": len(all_mses),
                "mse_min": float(min(all_mses)),
                "mse_max": float(max(all_mses)),
                "acc_min": float(min(all_accs)),
                "acc_max": float(max(all_accs)),
            })

            wandb.finish()
            print("uploaded to wandb")

        except Exception as e:
            print(f"wandb upload failed: {e}")


if __name__ == "__main__":
    main()
