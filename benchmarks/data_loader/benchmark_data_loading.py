#!/usr/bin/env python
"""Benchmark script comparing RoundRobin vs MultiViewTransform data loading approaches.

This script compares two approaches:
1. Round-robin: Uses RoundRobinMultiViewTransform with RepeatedRandomSampler (loads same image multiple times)
2. MultiViewTransform: Loads each image once, applies multiple transforms (more memory efficient)

The script automatically tests multiple batch sizes and shows performance metrics.

Usage:
    python examples/benchmark_data_loading.py
    python examples/benchmark_data_loading.py --batch-sizes 256 1024
"""

import time
import sys
from pathlib import Path
import argparse
import torch
import torch.utils.data
import numpy as np
from collections import defaultdict
import gc

# Add parent directory to path to import stable_pretraining
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.data.sampler import RepeatedRandomSampler
from utils import get_data_dir


def create_augmentation_pipeline():
    """Create a standard augmentation pipeline for benchmarking."""
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(p=1.0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def create_multiview_transform(n_views=2):
    """Create MultiViewTransform with n_views."""
    transform_list = [create_augmentation_pipeline() for _ in range(n_views)]
    return transforms.MultiViewTransform(transform_list)


def create_roundrobin_transform(n_views=2):
    """Create RoundRobinMultiViewTransform with n_views."""
    transform_list = [create_augmentation_pipeline() for _ in range(n_views)]
    return transforms.RoundRobinMultiViewTransform(transform_list)


def benchmark_multiview_approach(
    dataset_name="clane9/imagenet-100",
    batch_size=64,
    n_views=2,
    num_workers=4,
    num_iterations=50,
    device="cuda",
    verbose=True,
):
    """Benchmark the MultiViewTransform approach (used in benchmarks)."""
    if verbose:
        print("\n" + "=" * 60)
        print(f"MultiViewTransform - Batch Size: {batch_size}")
        print("  - Loads each image ONCE")
        print(f"  - Applies {n_views} transforms to create views")
        print("=" * 60)

    data_dir = get_data_dir("imagenet100")

    # Create dataset with MultiViewTransform
    dataset = spt.data.HFDataset(
        dataset_name,
        split="train",
        cache_dir=str(data_dir),
        transform=create_multiview_transform(n_views),
    )

    # Standard DataLoader with shuffle
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True,
    )

    return benchmark_dataloader(
        dataloader, num_iterations, device, n_views, verbose=verbose
    )


def benchmark_roundrobin_approach(
    dataset_name="clane9/imagenet-100",
    batch_size=64,
    n_views=2,
    num_workers=4,
    num_iterations=50,
    device="cuda",
    verbose=True,
):
    """Benchmark the RoundRobin approach with RepeatedRandomSampler."""
    if verbose:
        print("\n" + "=" * 60)
        print(f"RoundRobin - Batch Size: {batch_size}")
        print("  - Uses RepeatedRandomSampler to repeat indices")
        print("  - RoundRobinMultiViewTransform cycles through transforms")
        print("=" * 60)

    data_dir = get_data_dir("imagenet100")

    # Create dataset with RoundRobinMultiViewTransform
    dataset = spt.data.HFDataset(
        dataset_name,
        split="train",
        cache_dir=str(data_dir),
        transform=create_roundrobin_transform(n_views),
    )

    # Use RepeatedRandomSampler
    sampler = RepeatedRandomSampler(dataset, n_views=n_views)

    # DataLoader with custom sampler
    # Note: batch_size here is total augmented samples
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size * n_views,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
    )

    return benchmark_dataloader(
        dataloader, num_iterations, device, n_views, batch_size, verbose=verbose
    )


def benchmark_dataloader(
    dataloader,
    num_iterations,
    device,
    expected_views,
    unique_batch_size=None,
    verbose=True,
):
    """Common benchmarking logic for all dataloaders.

    Args:
        dataloader: The dataloader to benchmark
        num_iterations: Number of batches to test
        device: Device to test on (cuda/cpu)
        expected_views: Number of views expected (for MultiView approach)
        unique_batch_size: Actual number of unique images (for RepeatedSampler approaches)
        verbose: Whether to print progress
    """
    # Warmup
    if verbose:
        print("  Warming up...")
    dataloader_iter = iter(dataloader)
    for _ in range(min(5, num_iterations // 2)):
        batch = next(dataloader_iter)
        if device == "cuda":
            if expected_views > 1 and isinstance(batch, list):
                # MultiViewTransform returns list of view dicts
                for view in batch:
                    view["image"].to(device, non_blocking=True)
            elif expected_views > 1 and isinstance(batch.get("image"), list):
                # Alternative format: dict with list of images
                for img in batch["image"]:
                    img.to(device, non_blocking=True)
            else:
                batch["image"].to(device, non_blocking=True)

    # Reset iterator for actual benchmark
    dataloader_iter = iter(dataloader)

    # Metrics storage
    metrics = defaultdict(list)

    # Benchmark loop
    if verbose:
        print(f"  Running {num_iterations} iterations...")
    torch.cuda.synchronize() if device == "cuda" else None

    total_start = time.perf_counter()

    for i in range(num_iterations):
        # Time data loading
        iter_start = time.perf_counter()
        batch = next(dataloader_iter)
        load_time = time.perf_counter() - iter_start

        # Time GPU transfer
        transfer_start = time.perf_counter()
        if device == "cuda":
            if expected_views > 1 and isinstance(batch, list):
                # MultiViewTransform returns list of view dicts
                for view in batch:
                    view["image"].to(device, non_blocking=True)
                torch.cuda.synchronize()
                batch_size = batch[0]["image"].shape[0]  # Unique images
                total_samples = batch_size * len(batch)  # Total augmented samples
            elif expected_views > 1 and isinstance(batch.get("image"), list):
                # Alternative format: dict with list of images
                for img in batch["image"]:
                    img.to(device, non_blocking=True)
                torch.cuda.synchronize()
                batch_size = batch["image"][0].shape[0]
                total_samples = batch_size * len(batch["image"])
            else:
                # RepeatedSampler or single view
                batch["image"].to(device, non_blocking=True)
                torch.cuda.synchronize()
                if unique_batch_size is not None:
                    # For RepeatedSampler: we know the actual unique images
                    batch_size = unique_batch_size
                    total_samples = batch["image"].shape[
                        0
                    ]  # This is batch_size * n_views
                else:
                    batch_size = batch["image"].shape[0]
                    total_samples = batch_size
        else:
            if expected_views > 1 and isinstance(batch, list):
                batch_size = batch[0]["image"].shape[0]
                total_samples = batch_size * len(batch)
            elif expected_views > 1 and isinstance(batch.get("image"), list):
                batch_size = batch["image"][0].shape[0]
                total_samples = batch_size * len(batch["image"])
            else:
                if unique_batch_size is not None:
                    batch_size = unique_batch_size
                    total_samples = batch["image"].shape[0]
                else:
                    batch_size = batch["image"].shape[0]
                    total_samples = batch_size

        transfer_time = time.perf_counter() - transfer_start

        # Store metrics
        metrics["load_time"].append(load_time)
        metrics["transfer_time"].append(transfer_time)
        metrics["batch_size"].append(batch_size)
        metrics["total_samples"].append(total_samples)

        # Progress
        if verbose and (i + 1) % max(1, num_iterations // 5) == 0:
            print(f"    Iteration {i + 1}/{num_iterations}")

    total_time = time.perf_counter() - total_start

    # Calculate statistics
    results = {
        "total_time": total_time,
        "avg_load_time": np.mean(metrics["load_time"]),
        "std_load_time": np.std(metrics["load_time"]),
        "avg_transfer_time": np.mean(metrics["transfer_time"]),
        "std_transfer_time": np.std(metrics["transfer_time"]),
        "unique_images_per_second": np.sum(metrics["batch_size"]) / total_time,
        "avg_total_time": np.mean(metrics["load_time"])
        + np.mean(metrics["transfer_time"]),
    }

    # Print results
    if verbose:
        print("\n  Results:")
        print(
            f"    Load time: {results['avg_load_time'] * 1000:.2f}ms ± {results['std_load_time'] * 1000:.2f}ms"
        )
        print(
            f"    Transfer time: {results['avg_transfer_time'] * 1000:.2f}ms ± {results['std_transfer_time'] * 1000:.2f}ms"
        )
        print(f"    Total time per batch: {results['avg_total_time'] * 1000:.2f}ms")
        print(
            f"    Throughput: {results['unique_images_per_second']:.1f} unique images/s"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark data loading approaches")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[256, 1024],
        help="List of batch sizes to test (e.g., 256 1024)",
    )
    parser.add_argument(
        "--n-views", type=int, default=2, help="Number of augmented views per image"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of iterations to benchmark per batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to benchmark on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="clane9/imagenet-100",
        help="HuggingFace dataset to use",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Print detailed progress"
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("\n" + "=" * 70)
    print("DATA LOADING BENCHMARK: RoundRobin vs MultiViewTransform")
    print("=" * 70)
    print("Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch sizes to test: {args.batch_sizes}")
    print(f"  Views per image: {args.n_views}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Iterations per test: {args.num_iterations}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    multiview_results = []
    roundrobin_results = []

    # Test each batch size
    for batch_size in args.batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"TESTING BATCH SIZE: {batch_size}")
        print(f"{'=' * 70}")

        # Clean up between runs
        gc.collect()
        torch.cuda.empty_cache() if args.device == "cuda" else None

        # Benchmark MultiViewTransform
        mv_result = benchmark_multiview_approach(
            dataset_name=args.dataset,
            batch_size=batch_size,
            n_views=args.n_views,
            num_workers=args.num_workers,
            num_iterations=args.num_iterations,
            device=args.device,
            verbose=args.verbose,
        )
        multiview_results.append(mv_result)

        gc.collect()
        torch.cuda.empty_cache() if args.device == "cuda" else None

        # Benchmark RoundRobin
        rr_result = benchmark_roundrobin_approach(
            dataset_name=args.dataset,
            batch_size=batch_size,
            n_views=args.n_views,
            num_workers=args.num_workers,
            num_iterations=args.num_iterations,
            device=args.device,
            verbose=args.verbose,
        )
        roundrobin_results.append(rr_result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    print(
        f"{'Batch':<10} {'Approach':<20} {'Load (ms)':<20} {'Transfer (ms)':<20} {'Throughput (img/s)':<20}"
    )
    print("-" * 80)

    for i, batch_size in enumerate(args.batch_sizes):
        # MultiView results
        mv = multiview_results[i]
        print(
            f"{batch_size:<10} {'MultiViewTransform':<20} "
            f"{mv['avg_load_time'] * 1000:>8.2f} ± {mv['std_load_time'] * 1000:>4.2f}    "
            f"{mv['avg_transfer_time'] * 1000:>8.2f} ± {mv['std_transfer_time'] * 1000:>4.2f}    "
            f"{mv['unique_images_per_second']:>10.1f}"
        )

        # RoundRobin results
        rr = roundrobin_results[i]
        print(
            f"{batch_size:<10} {'RoundRobin':<20} "
            f"{rr['avg_load_time'] * 1000:>8.2f} ± {rr['std_load_time'] * 1000:>4.2f}    "
            f"{rr['avg_transfer_time'] * 1000:>8.2f} ± {rr['std_transfer_time'] * 1000:>4.2f}    "
            f"{rr['unique_images_per_second']:>10.1f}"
        )

        # Speedup
        speedup = mv["unique_images_per_second"] / rr["unique_images_per_second"]
        winner = "MultiView" if speedup > 1 else "RoundRobin"
        print(f"{'':10} {f'→ {winner} is {abs(speedup - 1) * 100:.1f}% faster':<20}")
        print("-" * 80)

    # Find overall winner
    avg_mv_throughput = np.mean(
        [r["unique_images_per_second"] for r in multiview_results]
    )
    avg_rr_throughput = np.mean(
        [r["unique_images_per_second"] for r in roundrobin_results]
    )

    print("\n" + "=" * 70)
    if avg_mv_throughput > avg_rr_throughput:
        speedup = avg_mv_throughput / avg_rr_throughput
        print("🏆 WINNER: MultiViewTransform")
        print(f"   Average {speedup:.2f}x faster across all batch sizes")
        print("   Better memory efficiency (loads each image once)")
    else:
        speedup = avg_rr_throughput / avg_mv_throughput
        print("🏆 WINNER: RoundRobin")
        print(f"   Average {speedup:.2f}x faster across all batch sizes")
    print("=" * 70)


if __name__ == "__main__":
    main()
