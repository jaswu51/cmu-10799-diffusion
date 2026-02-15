"""
Generate Reflow Pairs for Stage 2 (Rectified Flow)

Uses a trained Stage 1 model to generate synthetic (noise, image) pairs.
These pairs are saved to disk and used for Stage 2 (Reflow) training.

Usage:
    python generate_reflow_pairs.py \
        --checkpoint logs/rectified_flow_*/checkpoints/rectified_flow_final.pt \
        --num_pairs 60000 \
        --batch_size 256 \
        --num_steps 50 \
        --output_dir data/reflow_pairs
"""

import os
import argparse

import torch
from tqdm import tqdm

from src.models.dit import create_dit_from_config
from src.models import create_model_from_config
from src.methods.rectified_flow import RectifiedFlow
from src.utils import EMA


def main():
    parser = argparse.ArgumentParser(description="Generate reflow pairs")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Stage 1 checkpoint path"
    )
    parser.add_argument(
        "--num_pairs", type=int, default=60000, help="Number of pairs to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for generation"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Euler steps for ODE integration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reflow_pairs",
        help="Directory to save pairs",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    # Create model (handle both UNet and DiT)
    model_type = config.get("model", {}).get("type", "unet")
    if model_type == "dit":
        model = create_dit_from_config(config).to(device)
    else:
        model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    # Apply EMA weights
    ema = EMA(model, decay=config["training"]["ema_decay"])
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
        ema.apply_shadow()
        print("Using EMA weights")

    # Create method
    method = RectifiedFlow.from_config(model, config, device)
    method.eval_mode()

    # Image shape
    data_config = config["data"]
    image_shape = (
        data_config["channels"],
        data_config["image_size"],
        data_config["image_size"],
    )

    # Generate pairs
    os.makedirs(args.output_dir, exist_ok=True)

    all_x0 = []
    all_x1 = []
    remaining = args.num_pairs

    print(f"Generating {args.num_pairs} reflow pairs ({args.num_steps} Euler steps)...")
    pbar = tqdm(total=args.num_pairs, desc="Generating pairs")

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        x_0, x_1_hat = method.generate_reflow_pairs(
            batch_size=batch_size,
            image_shape=image_shape,
            num_steps=args.num_steps,
        )
        all_x0.append(x_0)
        all_x1.append(x_1_hat)
        remaining -= batch_size
        pbar.update(batch_size)

    pbar.close()

    # Concatenate and save
    all_x0 = torch.cat(all_x0, dim=0)[: args.num_pairs]
    all_x1 = torch.cat(all_x1, dim=0)[: args.num_pairs]

    x0_path = os.path.join(args.output_dir, "x0_noise.pt")
    x1_path = os.path.join(args.output_dir, "x1_generated.pt")

    print(f"Saving x0 (noise): {all_x0.shape} -> {x0_path}")
    torch.save(all_x0, x0_path)

    print(f"Saving x1 (generated): {all_x1.shape} -> {x1_path}")
    torch.save(all_x1, x1_path)

    print(f"Done! Saved {args.num_pairs} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
