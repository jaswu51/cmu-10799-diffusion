"""
Trajectory Visualization for Rectified Flow

Visualizes ODE trajectories to compare 1-Rectified Flow vs 2-Rectified Flow (Reflow).
Includes:
1. Trajectory straightness metric S
2. PCA-projected 2D trajectory plots
3. Pixel-space interpolation grids

Usage:
    # Single model visualization
    python visualize_trajectories.py \
        --checkpoint logs/rectified_flow_*/checkpoints/rectified_flow_final.pt \
        --method rectified_flow \
        --num_trajectories 20 \
        --output_dir visualizations

    # Compare 1-RF vs 2-RF
    python visualize_trajectories.py \
        --checkpoint logs/stage1/rectified_flow_final.pt \
        --checkpoint2 logs/stage2/reflow_final.pt \
        --method rectified_flow \
        --num_trajectories 20 \
        --output_dir visualizations
"""

import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

from src.models import create_model_from_config
from src.models.dit import create_dit_from_config
from src.data import unnormalize
from src.methods.rectified_flow import RectifiedFlow
from src.utils import EMA


def load_model(checkpoint_path, device):
    """Load model from checkpoint and apply EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_type = config.get("model", {}).get("type", "unet")
    if model_type == "dit":
        model = create_dit_from_config(config).to(device)
    else:
        model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    ema = EMA(model, decay=config["training"]["ema_decay"])
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
        ema.apply_shadow()

    model.eval()
    return model, config


@torch.no_grad()
def collect_trajectories(model, device, image_shape, num_trajectories=20, num_steps=50):
    """
    Run ODE integration and record the full trajectory at each step.

    Returns:
        trajectories: (num_trajectories, num_steps+1, C*H*W) flattened trajectory points
        t_vals: (num_steps+1,) time values
    """
    C, H, W = image_shape
    dim = C * H * W

    x = torch.randn(num_trajectories, C, H, W, device=device)
    t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    dt = t_vals[1] - t_vals[0]

    # Store trajectory: shape (num_traj, num_steps+1, dim)
    trajectories = torch.zeros(num_trajectories, num_steps + 1, dim)
    trajectories[:, 0] = x.view(num_trajectories, -1).cpu()

    for i in range(num_steps):
        t = torch.full((num_trajectories,), t_vals[i], device=device)
        v = model(x, t)
        x = x + v * dt
        trajectories[:, i + 1] = x.view(num_trajectories, -1).cpu()

    return trajectories, t_vals.cpu()


def compute_straightness(trajectories):
    """
    Compute trajectory straightness S for each trajectory.

    S = ||x_1 - x_0|| / sum(||x_{t+1} - x_t||)
    S = 1 means perfectly straight.

    Args:
        trajectories: (N, T+1, D) trajectory points

    Returns:
        straightness: (N,) straightness values
    """
    # Direct distance: ||x_1 - x_0||
    direct = torch.norm(trajectories[:, -1] - trajectories[:, 0], dim=-1)

    # Path length: sum of step distances
    diffs = trajectories[:, 1:] - trajectories[:, :-1]  # (N, T, D)
    step_lengths = torch.norm(diffs, dim=-1)  # (N, T)
    path_length = step_lengths.sum(dim=-1)  # (N,)

    straightness = direct / (path_length + 1e-8)
    return straightness


def plot_pca_trajectories(
    trajectories,
    t_vals,
    title="ODE Trajectories (PCA projection)",
    ax=None,
    color_by_time=True,
):
    """
    Project high-dimensional trajectories to 2D via PCA and plot.

    Args:
        trajectories: (N, T+1, D) trajectory points
        t_vals: (T+1,) time values
        title: plot title
        ax: matplotlib axis (created if None)
        color_by_time: if True, color segments by time t
    """
    N, T_plus_1, D = trajectories.shape

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Fit PCA on all trajectory points
    from sklearn.decomposition import PCA
    all_points = trajectories.reshape(-1, D).numpy()
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_points)
    all_2d = all_2d.reshape(N, T_plus_1, 2)

    # Color map for time
    cmap = plt.cm.viridis

    for i in range(N):
        points = all_2d[i]  # (T+1, 2)

        if color_by_time:
            # Create colored line segments
            segments = np.array(
                [[points[j], points[j + 1]] for j in range(T_plus_1 - 1)]
            )
            t_colors = t_vals[:-1].numpy()
            lc = LineCollection(segments, cmap=cmap, alpha=0.6, linewidths=1.0)
            lc.set_array(t_colors)
            ax.add_collection(lc)
        else:
            ax.plot(points[:, 0], points[:, 1], alpha=0.4, linewidth=0.8)

        # Mark start (noise) and end (data)
        ax.scatter(points[0, 0], points[0, 1], c="blue", s=15, zorder=5, alpha=0.5)
        ax.scatter(points[-1, 0], points[-1, 1], c="red", s=15, zorder=5, alpha=0.5)

        # Draw straight line for reference
        ax.plot(
            [points[0, 0], points[-1, 0]],
            [points[0, 1], points[-1, 1]],
            "k--",
            alpha=0.1,
            linewidth=0.5,
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")

    # Legend
    ax.scatter([], [], c="blue", s=30, label="Noise (t=0)")
    ax.scatter([], [], c="red", s=30, label="Data (t=1)")
    ax.plot([], [], "k--", alpha=0.3, label="Straight line")
    ax.legend(loc="upper right", fontsize=8)

    return ax


def plot_straightness_histogram(
    straightness_list, labels, output_path=None
):
    """Plot histogram of trajectory straightness values."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ["#4C72B0", "#DD8452"]
    for i, (s, label) in enumerate(zip(straightness_list, labels)):
        s_np = s.numpy()
        ax.hist(
            s_np,
            bins=30,
            alpha=0.6,
            label=f"{label} (mean={s_np.mean():.4f})",
            color=colors[i % len(colors)],
            edgecolor="white",
        )

    ax.set_xlabel("Straightness S (1.0 = perfectly straight)")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory Straightness Distribution")
    ax.legend()
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved straightness histogram to {output_path}")
    return fig


def plot_interpolation_grid(trajectories, image_shape, num_show=4, num_time_points=8, output_path=None):
    """
    Show pixel-space interpolation: x_t at different time steps.

    Args:
        trajectories: (N, T+1, D) trajectory points
        image_shape: (C, H, W)
        num_show: number of trajectories to show
        num_time_points: number of time points to display
    """
    N, T_plus_1, D = trajectories.shape
    C, H, W = image_shape
    num_show = min(num_show, N)

    # Select evenly spaced time indices
    time_indices = np.linspace(0, T_plus_1 - 1, num_time_points).astype(int)

    fig, axes = plt.subplots(num_show, num_time_points, figsize=(2 * num_time_points, 2 * num_show))
    if num_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_show):
        for j, t_idx in enumerate(time_indices):
            img = trajectories[i, t_idx].reshape(C, H, W)
            img = unnormalize(img.unsqueeze(0)).squeeze(0)
            img = img.permute(1, 2, 0).numpy().clip(0, 1)

            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if i == 0:
                t_val = t_idx / (T_plus_1 - 1)
                axes[i, j].set_title(f"t={t_val:.2f}", fontsize=9)

    plt.suptitle("Pixel-space Trajectory (noise -> data)", fontsize=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved interpolation grid to {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize RF trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model 1 checkpoint")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Model 2 checkpoint (for comparison)")
    parser.add_argument("--label1", type=str, default="1-Rectified Flow", help="Label for model 1")
    parser.add_argument("--label2", type=str, default="2-Rectified Flow (Reflow)", help="Label for model 2")
    parser.add_argument("--method", type=str, default="rectified_flow")
    parser.add_argument("--num_trajectories", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model 1
    print(f"Loading model 1: {args.checkpoint}")
    model1, config1 = load_model(args.checkpoint, device)
    image_shape = (
        config1["data"]["channels"],
        config1["data"]["image_size"],
        config1["data"]["image_size"],
    )

    # Collect trajectories for model 1
    print(f"Collecting {args.num_trajectories} trajectories ({args.num_steps} steps)...")
    traj1 = collect_trajectories(
        model1, device, image_shape, args.num_trajectories, args.num_steps
    )
    trajs1, t_vals = traj1
    s1 = compute_straightness(trajs1)
    print(f"  {args.label1}: straightness mean={s1.mean():.4f}, std={s1.std():.4f}")

    # Optionally load model 2
    trajs2 = None
    s2 = None
    if args.checkpoint2:
        print(f"Loading model 2: {args.checkpoint2}")
        model2, config2 = load_model(args.checkpoint2, device)

        # Use same noise for fair comparison
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        print(f"Collecting {args.num_trajectories} trajectories ({args.num_steps} steps)...")
        trajs2, t_vals2 = collect_trajectories(
            model2, device, image_shape, args.num_trajectories, args.num_steps
        )
        s2 = compute_straightness(trajs2)
        print(f"  {args.label2}: straightness mean={s2.mean():.4f}, std={s2.std():.4f}")

    # === Plot 1: PCA Trajectories ===
    if args.checkpoint2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_pca_trajectories(trajs1, t_vals, title=f"{args.label1}\nS={s1.mean():.4f}", ax=ax1)
        plot_pca_trajectories(trajs2, t_vals, title=f"{args.label2}\nS={s2.mean():.4f}", ax=ax2)
        plt.suptitle("PCA-Projected ODE Trajectories", fontsize=14)
        plt.tight_layout()
        pca_path = os.path.join(args.output_dir, "pca_trajectories_comparison.png")
        fig.savefig(pca_path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA comparison to {pca_path}")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_pca_trajectories(trajs1, t_vals, title=f"{args.label1}\nS={s1.mean():.4f}", ax=ax)
        plt.tight_layout()
        pca_path = os.path.join(args.output_dir, "pca_trajectories.png")
        fig.savefig(pca_path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA plot to {pca_path}")
    plt.close()

    # === Plot 2: Straightness Histogram ===
    if args.checkpoint2:
        plot_straightness_histogram(
            [s1, s2],
            [args.label1, args.label2],
            output_path=os.path.join(args.output_dir, "straightness_histogram.png"),
        )
    else:
        plot_straightness_histogram(
            [s1],
            [args.label1],
            output_path=os.path.join(args.output_dir, "straightness_histogram.png"),
        )
    plt.close()

    # === Plot 3: Pixel-space Interpolation ===
    plot_interpolation_grid(
        trajs1,
        image_shape,
        num_show=4,
        num_time_points=8,
        output_path=os.path.join(args.output_dir, f"interpolation_{args.label1.replace(' ', '_')}.png"),
    )
    plt.close()

    if trajs2 is not None:
        plot_interpolation_grid(
            trajs2,
            image_shape,
            num_show=4,
            num_time_points=8,
            output_path=os.path.join(args.output_dir, f"interpolation_{args.label2.replace(' ', '_')}.png"),
        )
        plt.close()

    # === Save straightness values ===
    stats = {args.label1: {"mean": s1.mean().item(), "std": s1.std().item()}}
    if s2 is not None:
        stats[args.label2] = {"mean": s2.mean().item(), "std": s2.std().item()}

    stats_path = os.path.join(args.output_dir, "straightness_stats.txt")
    with open(stats_path, "w") as f:
        for label, vals in stats.items():
            f.write(f"{label}: S = {vals['mean']:.6f} +/- {vals['std']:.6f}\n")
    print(f"Saved straightness stats to {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
