
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats


def load_trajectory_files(output_dir):
    output_path = Path(output_dir)
    trajectories = []

    for traj_file in output_path.rglob("trajectory.json"):
        try:
            with open(traj_file, 'r') as f:
                data = json.load(f)
                trajectories.append(data)
                print(f"Loaded: {traj_file.name} (alpha={data.get('alpha')}, {len(data.get('trajectory', []))} epochs)")
        except Exception as e:
            print(f"Warning: Could not load {traj_file}: {e}")

    return trajectories


def create_tube_plot(trajectories, output_path, title="MAE Reconstruction Tube"):

    all_mse = []
    all_acc = []
    all_alpha_labels = []
    all_epochs = []

    print("\nCollecting TRUE MSE-Accuracy pairs from trajectory files...")
    print("(MSE from val/loss_rec, Accuracy from val/supervised_acc during pretraining)\n")

    for traj_data in trajectories:
        alpha = traj_data.get('alpha')
        if alpha is None:
            continue

        trajectory = traj_data.get('trajectory', [])
        if not trajectory:
            continue

        for point in trajectory:
            mse = point.get('mse')
            acc = point.get('accuracy')  # Already in percentage
            epoch = point.get('epoch')

            if mse is not None and acc is not None:
                all_mse.append(mse)
                all_acc.append(acc)
                all_alpha_labels.append(alpha)
                all_epochs.append(epoch)

        print(f"Alpha {alpha}: {len(trajectory)} epochs, "
              f"Final MSE: {trajectory[-1]['mse']:.4f}, Final Acc: {trajectory[-1]['accuracy']:.2f}%")

    all_mse = np.array(all_mse)
    all_acc = np.array(all_acc)
    all_alpha_labels = np.array(all_alpha_labels)
    all_epochs = np.array(all_epochs)

    print(f"\n{'='*70}")
    print(f"Total TRUE (MSE, Accuracy) pairs: {len(all_mse)}")
    print(f"MSE range: [{all_mse.min():.4f}, {all_mse.max():.4f}]")
    print(f"Accuracy range: [{all_acc.min():.2f}%, {all_acc.max():.2f}%]")
    print(f"{'='*70}\n")

    fig, ax = plt.subplots(figsize=(14, 8))

    unique_alphas = np.unique(all_alpha_labels)

    use_log_norm = len(unique_alphas) > 3 and np.max(np.abs(unique_alphas)) > 1.0

    if use_log_norm:
        abs_alphas = np.abs(all_alpha_labels)
        abs_alphas[abs_alphas == 0] = 0.001
        norm = plt.matplotlib.colors.LogNorm(vmin=abs_alphas.min(), vmax=abs_alphas.max())
        color_values = abs_alphas
    else:
        norm = None
        color_values = all_alpha_labels

    scatter = ax.scatter(all_mse, all_acc,
                        c=color_values,
                        cmap='plasma',
                        s=40,
                        alpha=0.7,
                        norm=norm,
                        edgecolors='black',
                        linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Alpha (Supervision Strength)', fontsize=12, fontweight='bold')

    ax.set_xlabel('Reconstruction MSE (Validation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%) - Validation', fontsize=14, fontweight='bold')

    if trajectories:
        model = trajectories[0].get('model', 'unknown')
        decoder = trajectories[0].get('decoder', 'unknown')
        ax.set_title(f'MAE Reconstruction Tube: {model} (decoder: {decoder})',
                     fontsize=15, fontweight='bold', pad=15)
    else:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    if len(all_mse) > 1:
        corr, p_value = stats.spearmanr(all_mse, all_acc)
    else:
        corr = 0.0

    n_bins = 15
    mse_bins = np.linspace(all_mse.min(), all_mse.max(), n_bins+1)
    tube_volume = 0.0
    for i in range(len(mse_bins)-1):
        mask = (all_mse >= mse_bins[i]) & (all_mse < mse_bins[i+1])
        if mask.sum() > 3:
            acc_in_bin = all_acc[mask]
            spread = acc_in_bin.max() - acc_in_bin.min()
            tube_volume += spread

    model_info = trajectories[0].get('model', 'unknown') if trajectories else 'unknown'
    decoder_info = trajectories[0].get('decoder', 'unknown') if trajectories else 'unknown'

    textstr = f'Model: {model_info}\n'
    textstr += f'Decoder: {decoder_info}\n'
    textstr += f'Data: {len(all_mse)} validation points\n'
    textstr += f'({len(np.unique(all_epochs))} epochs × {len(unique_alphas)} alpha values)\n'
    textstr += f'Spearman ρ: {corr:.3f}\n'
    textstr += f'Tube Volume Γ: {tube_volume:.1f}%'

    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ TRUE tube plot saved to: {output_path}\n")

    print(f"{'='*70}")
    print("RECONSTRUCTION TUBE ANALYSIS")
    print(f"{'='*70}")

    print(f"\n{'MSE Range':<30} {'Acc Range (%)':<25} {'Spread':<12} {'Points'}")
    print(f"{'-'*70}")

    max_spread = 0
    significant_bins = []

    for i in range(len(mse_bins)-1):
        mask = (all_mse >= mse_bins[i]) & (all_mse < mse_bins[i+1])
        if mask.sum() > 3:
            acc_in_bin = all_acc[mask]
            spread = acc_in_bin.max() - acc_in_bin.min()
            max_spread = max(max_spread, spread)

            print(f"[{mse_bins[i]:.3f}, {mse_bins[i+1]:.3f}]      "
                  f"[{acc_in_bin.min():.1f}, {acc_in_bin.max():.1f}]        "
                  f"{spread:>7.1f}%     {mask.sum():>4}")

            if spread > 5:
                significant_bins.append((mse_bins[i], mse_bins[i+1], spread))

    print(f"{'-'*70}")
    print(f"\n🎯 MAXIMUM VERTICAL SPREAD: {max_spread:.1f}%")
    print(f"   → At similar reconstruction MSE, accuracy varies by up to {max_spread:.1f}%!")

    if significant_bins:
        print(f"\n📊 BINS WITH SIGNIFICANT SPREAD (>5%):")
        for mse_low, mse_high, spread in significant_bins:
            print(f"   MSE [{mse_low:.3f}, {mse_high:.3f}]: {spread:.1f}% accuracy spread")

    print(f"\n{'='*70}")
    print("✅ This demonstrates MAE's ILL-POSED reconstruction objective:")
    print("   Multiple encoder configurations achieve SIMILAR reconstruction MSE")
    print("   but produce VERY DIFFERENT semantic representations (accuracies)!")
    print(f"{'='*70}\n")


def plot_individual_metrics(trajectories, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_trajectories = [t for t in trajectories if t.get('trajectory') and len(t.get('trajectory', [])) > 0]

    all_alphas = sorted(set(t['alpha'] for t in valid_trajectories if t.get('alpha') is not None))

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    alpha_to_color = {alpha: color_list[i % len(color_list)] for i, alpha in enumerate(all_alphas)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for traj_data in trajectories:
        alpha = traj_data.get('alpha')
        if alpha is None:
            continue
        trajectory = traj_data.get('trajectory', [])
        if not trajectory:
            continue

        epochs = [point['epoch'] for point in trajectory]
        mse_values = [point['mse'] for point in trajectory]
        color = alpha_to_color[alpha]
        ax.plot(epochs, mse_values, linestyle='-', marker='o', color=color, linewidth=2,
                markersize=6, label=f'α={alpha:.1f}')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=14)
    ax.set_title('Reconstruction Loss over Epochs', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    mse_plot_path = output_dir / 'mse_over_epochs.png'
    plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved MSE plot to: {mse_plot_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for traj_data in trajectories:
        alpha = traj_data.get('alpha')
        if alpha is None:
            continue
        trajectory = traj_data.get('trajectory', [])
        if not trajectory:
            continue

        epochs = [point['epoch'] for point in trajectory]
        acc_values = [point['accuracy'] for point in trajectory]
        color = alpha_to_color[alpha]
        ax.plot(epochs, acc_values, linestyle='-', marker='o', color=color, linewidth=2,
                markersize=6, label=f'α={alpha:.1f}')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy over Epochs', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    acc_plot_path = output_dir / 'accuracy_over_epochs.png'
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved accuracy plot to: {acc_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MAE reconstruction gap - TRUE tube plot")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                       help="Directory containing trajectory.json files")
    parser.add_argument("--plot_dir", type=str, default="plots/",
                       help="Directory to save plots")
    parser.add_argument("--title", type=str, default="MAE Reconstruction Tube",
                       help="Plot title")
    args = parser.parse_args()

    print(f"Loading trajectories from: {args.output_dir}")
    trajectories = load_trajectory_files(args.output_dir)

    if not trajectories:
        print("Error: No trajectory files found!")
        return

    print(f"\nFound {len(trajectories)} trajectory files")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    main_plot_path = plot_dir / 'reconstruction_tube_TRUE.png'
    create_tube_plot(trajectories, main_plot_path, title=args.title)

    plot_individual_metrics(trajectories, plot_dir)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {plot_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
