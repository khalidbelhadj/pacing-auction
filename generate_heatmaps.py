#!/usr/bin/env python3
"""
Script to run the heatmap performance test with various options.
This allows for more control over the test range and parameters.
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Dict, Any, Union, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import itertools

# Add project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pacing_auction.auction import Auction
from pacing_auction.data import PNE, Cycle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("heatmap_test")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run pacing auction heatmap performance tests"
    )

    parser.add_argument(
        "--min-size", type=int, default=2, help="Minimum value for n and m (default: 2)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=15,
        help="Maximum value for n and m (default: 15)",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Step size between values (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs for each configuration (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_heatmap",
        help="Directory to save results (default: 'results_heatmap')",
    )
    parser.add_argument(
        "--no-threading",
        action="store_true",
        help="Disable threading in the auction algorithm",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum time (in seconds) to allow for a single configuration (default: 300)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help=f"Number of worker processes for parallel execution (default: {max(1, multiprocessing.cpu_count() - 1)} - autodetected)",
    )
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable multiprocessing and run tests sequentially",
    )

    return parser.parse_args()


def create_output_dir(output_dir: str) -> str:
    """Create output directory with timestamp"""

    # Create base directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir)

    return run_dir


def run_single_test(
    n: int, m: int, seed: int, no_threading: bool, timeout: int
) -> Dict[str, Any]:
    """
    Run a single test for a given configuration

    Parameters:
        n: Number of bidders
        m: Number of auctions
        seed: Random seed
        no_threading: Whether to disable threading
        timeout: Maximum time (in seconds) to allow for the test

    Returns:
        Dictionary with test results
    """
    result_data = {
        "runtime": -1,
        "iterations": -1,
        "result_type": "Error",
        "timeout": False,
        "error": False,
    }

    # Record start time
    start_time = time.perf_counter()

    try:
        # Create and run auction with timeout handling
        auction = Auction(
            n=n, m=m, seed=seed, collect_stats=True, threaded=not no_threading
        )

        # Use a timeout mechanism
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Auction run timed out after {timeout} seconds")

        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        # Run the auction
        result = auction.responses()

        # Clear the alarm
        signal.alarm(0)

        # Record end time
        exec_time = time.perf_counter() - start_time

        # Store results
        result_data["runtime"] = exec_time
        result_data["iterations"] = result.iteration
        result_data["result_type"] = type(result).__name__

    except TimeoutError:
        logger.warning(f"Configuration ({n}, {m}) timed out after {timeout} seconds")
        result_data["runtime"] = timeout
        result_data["timeout"] = True
        result_data["result_type"] = "Timeout"

    except Exception as e:
        logger.error(f"Error running configuration ({n}, {m}): {str(e)}")
        result_data["error"] = True

    return result_data


def process_configuration(
    config: Tuple[int, int, int, int, int, bool, int, int],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Process a single configuration with multiple runs

    Parameters:
        config: Tuple containing (i, j, n, m, seed, no_threading, timeout, runs)

    Returns:
        Tuple of (i, j, results)
    """
    i, j, n, m, base_seed, no_threading, timeout, runs = config

    config_results = []

    for run_idx in range(runs):
        # Use a different seed for each run
        run_seed = base_seed + run_idx

        # Run the test
        result = run_single_test(n, m, run_seed, no_threading, timeout)
        config_results.append(result)

    return (i, j, config_results)


def run_heatmap_test(
    min_size: int,
    max_size: int,
    step: int,
    seed: int,
    runs: int,
    output_dir: str,
    no_threading: bool,
    timeout: int,
    workers: int,
    no_multiprocessing: bool,
) -> Tuple[Dict[str, np.ndarray], List[List[List[Dict[str, Any]]]]]:
    """
    Run performance tests for n and m values in the specified range

    Parameters:
        min_size: Minimum value for n and m
        max_size: Maximum value for n and m
        step: Step size between values
        seed: Random seed base (will be incremented for each run)
        runs: Number of runs for each configuration
        output_dir: Directory to save results
        no_threading: Whether to disable threading
        timeout: Maximum time (in seconds) to allow for a single configuration
        workers: Number of worker processes to use
        no_multiprocessing: Whether to disable multiprocessing

    Returns:
        Tuple of (result_matrices, all_results)
    """
    # Generate the range of sizes
    sizes = list(range(min_size, max_size + 1, step))
    n_sizes = len(sizes)

    # Create matrices for storing aggregated results
    result_matrices = {
        "mean_runtime": np.zeros((n_sizes, n_sizes)),
        "median_runtime": np.zeros((n_sizes, n_sizes)),
        "min_runtime": np.zeros((n_sizes, n_sizes)),
        "max_runtime": np.zeros((n_sizes, n_sizes)),
        "mean_iterations": np.zeros((n_sizes, n_sizes)),
        "median_iterations": np.zeros((n_sizes, n_sizes)),
        "min_iterations": np.zeros((n_sizes, n_sizes)),
        "max_iterations": np.zeros((n_sizes, n_sizes)),
        "pne_percentage": np.zeros((n_sizes, n_sizes)),
        "cycle_percentage": np.zeros((n_sizes, n_sizes)),
        "timeout_percentage": np.zeros((n_sizes, n_sizes)),
        "error_percentage": np.zeros((n_sizes, n_sizes)),
        "timeouts": np.zeros((n_sizes, n_sizes), dtype=bool),
    }

    # Store all individual results
    all_results = [[[] for _ in range(n_sizes)] for _ in range(n_sizes)]

    total_configs = n_sizes * n_sizes
    total_tests = total_configs * runs
    completed = 0

    start_time = time.time()

    # Create a list of configurations to test
    configs = []
    for i, n in enumerate(sizes):
        for j, m in enumerate(sizes):
            configs.append((i, j, n, m, seed, no_threading, timeout, runs))

    if no_multiprocessing:
        # Process configurations sequentially
        logger.info(f"Running tests sequentially (multiprocessing disabled)")

        for config in configs:
            i, j, n, m = config[0], config[1], config[2], config[3]
            logger.info(
                f"Testing configuration ({n}, {m}) - {completed//runs}/{total_configs} configurations completed"
            )

            i, j, config_results = process_configuration(config)

            # Store results
            all_results[i][j] = config_results
            process_results(result_matrices, i, j, config_results, runs)

            completed += runs
            logger.info(
                f"Progress: {completed}/{total_tests} tests completed ({completed/total_tests*100:.1f}%)"
            )

            # Generate intermediate heatmaps every 5 configurations
            if (completed % (5 * runs) == 0) or (completed == total_tests):
                create_heatmaps(sizes, result_matrices, output_dir)
    else:
        # Process configurations in parallel using ProcessPoolExecutor
        logger.info(f"Running tests with {workers} worker processes")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all configurations to the executor
            future_to_config = {
                executor.submit(process_configuration, config): config
                for config in configs
            }

            # Process results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                i, j, n, m = config[0], config[1], config[2], config[3]

                try:
                    i, j, config_results = future.result()

                    # Store results
                    all_results[i][j] = config_results
                    process_results(result_matrices, i, j, config_results, runs)

                    completed += runs
                    logger.info(
                        f"Completed configuration ({n}, {m}) - Progress: {completed}/{total_tests} tests ({completed/total_tests*100:.1f}%)"
                    )

                    # Generate intermediate heatmaps every 5 configurations
                    if (completed % (5 * runs) == 0) or (completed == total_tests):
                        create_heatmaps(sizes, result_matrices, output_dir)

                except Exception as e:
                    logger.error(f"Error processing configuration ({n}, {m}): {str(e)}")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    return result_matrices, all_results


def process_results(
    result_matrices: Dict[str, np.ndarray],
    i: int,
    j: int,
    config_results: List[Dict[str, Any]],
    runs: int,
) -> None:
    """
    Process the results of a configuration and update the result matrices

    Parameters:
        result_matrices: Dictionary of result matrices to update
        i: Row index
        j: Column index
        config_results: List of test results for the configuration
        runs: Total number of runs
    """
    # Extract runtime and iteration data from valid results
    config_runtimes = []
    config_iterations = []
    timeout_count = 0
    error_count = 0
    pne_count = 0
    cycle_count = 0

    for result in config_results:
        if result["timeout"]:
            timeout_count += 1
        elif result["error"]:
            error_count += 1
        else:
            config_runtimes.append(result["runtime"])
            config_iterations.append(result["iterations"])

            if result["result_type"] == "PNE":
                pne_count += 1
            elif result["result_type"] == "Cycle":
                cycle_count += 1

    # Calculate statistics if there are valid results
    if config_runtimes:  # Only calculate if not empty
        result_matrices["mean_runtime"][i, j] = np.mean(config_runtimes)
        result_matrices["median_runtime"][i, j] = np.median(config_runtimes)
        result_matrices["min_runtime"][i, j] = np.min(config_runtimes)
        result_matrices["max_runtime"][i, j] = np.max(config_runtimes)
    else:
        # Mark as invalid
        result_matrices["timeouts"][i, j] = True

    if config_iterations:  # Only calculate if not empty
        result_matrices["mean_iterations"][i, j] = np.mean(config_iterations)
        result_matrices["median_iterations"][i, j] = np.median(config_iterations)
        result_matrices["min_iterations"][i, j] = np.min(config_iterations)
        result_matrices["max_iterations"][i, j] = np.max(config_iterations)

    # Calculate percentages
    valid_runs = runs - timeout_count - error_count
    result_matrices["pne_percentage"][i, j] = (
        (pne_count / runs) * 100 if runs > 0 else 0
    )
    result_matrices["cycle_percentage"][i, j] = (
        (cycle_count / runs) * 100 if runs > 0 else 0
    )
    result_matrices["timeout_percentage"][i, j] = (
        (timeout_count / runs) * 100 if runs > 0 else 0
    )
    result_matrices["error_percentage"][i, j] = (
        (error_count / runs) * 100 if runs > 0 else 0
    )

    # If all runs timed out or errored, mark this cell
    if valid_runs == 0:
        result_matrices["timeouts"][i, j] = True


def create_heatmaps(
    sizes: List[int],
    result_matrices: Dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """
    Create heatmaps for the test results

    Parameters:
        sizes: List of n and m values used
        result_matrices: Dictionary containing all result matrices
        output_dir: Directory to save the heatmap images
    """
    # Convert sizes to strings for tick labels
    size_labels = [str(s) for s in sizes]

    # Get timeouts matrix for masking
    timeouts = result_matrices["timeouts"]

    # Create mean runtime heatmap
    plt.figure(figsize=(12, 10))
    masked_mean_runtime = np.ma.masked_where(timeouts, result_matrices["mean_runtime"])
    ax1 = sns.heatmap(
        masked_mean_runtime,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("Mean Execution Time (seconds)", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_mean_runtime.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create median runtime heatmap
    plt.figure(figsize=(12, 10))
    masked_median_runtime = np.ma.masked_where(
        timeouts, result_matrices["median_runtime"]
    )
    ax2 = sns.heatmap(
        masked_median_runtime,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("Median Execution Time (seconds)", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_median_runtime.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create mean iterations heatmap
    plt.figure(figsize=(12, 10))
    masked_mean_iterations = np.ma.masked_where(
        timeouts, result_matrices["mean_iterations"]
    )
    ax3 = sns.heatmap(
        masked_mean_iterations,
        annot=True,
        fmt=".1f",
        cmap="plasma",
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("Mean Number of Iterations", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_mean_iterations.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create median iterations heatmap
    plt.figure(figsize=(12, 10))
    masked_median_iterations = np.ma.masked_where(
        timeouts, result_matrices["median_iterations"]
    )
    ax4 = sns.heatmap(
        masked_median_iterations,
        annot=True,
        fmt=".1f",
        cmap="plasma",
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("Median Number of Iterations", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_median_iterations.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create PNE percentage heatmap
    plt.figure(figsize=(12, 10))
    ax5 = sns.heatmap(
        result_matrices["pne_percentage"],
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("PNE Percentage (%)", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_pne_percentage.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create Cycle percentage heatmap
    plt.figure(figsize=(12, 10))
    ax6 = sns.heatmap(
        result_matrices["cycle_percentage"],
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("Cycle Percentage (%)", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_cycle_percentage.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create timeout percentage heatmap
    plt.figure(figsize=(12, 10))
    ax7 = sns.heatmap(
        result_matrices["timeout_percentage"],
        annot=True,
        fmt=".1f",
        cmap="Oranges",
        vmin=0,
        vmax=100,
        xticklabels=size_labels,
        yticklabels=size_labels,
    )
    plt.title("Timeout Percentage (%)", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_timeout_percentage.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create a combined result type heatmap
    plt.figure(figsize=(15, 12))

    # Create a combined matrix where the dominant result type is shown
    combined_result = np.zeros_like(result_matrices["pne_percentage"])

    for i in range(combined_result.shape[0]):
        for j in range(combined_result.shape[1]):
            pne_pct = result_matrices["pne_percentage"][i, j]
            cycle_pct = result_matrices["cycle_percentage"][i, j]
            timeout_pct = result_matrices["timeout_percentage"][i, j]
            error_pct = result_matrices["error_percentage"][i, j]

            # Find the dominant result type (highest percentage)
            max_pct = max(pne_pct, cycle_pct, timeout_pct, error_pct)

            if max_pct == pne_pct:
                combined_result[i, j] = 0  # PNE
            elif max_pct == cycle_pct:
                combined_result[i, j] = 1  # Cycle
            elif max_pct == timeout_pct:
                combined_result[i, j] = 2  # Timeout
            else:
                combined_result[i, j] = 3  # Error

    # Use a custom colormap for result types
    cmap = ListedColormap(["#66b3ff", "#ff9999", "#ffcc99", "#cccccc"])

    ax8 = sns.heatmap(
        combined_result,
        cmap=cmap,
        cbar=False,
        xticklabels=size_labels,
        yticklabels=size_labels,
    )

    # Add percentage text annotations
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            pne_pct = result_matrices["pne_percentage"][i, j]
            cycle_pct = result_matrices["cycle_percentage"][i, j]
            timeout_pct = result_matrices["timeout_percentage"][i, j]

            # Format the percentages
            text = f"PNE: {pne_pct:.0f}%\nCycle: {cycle_pct:.0f}%"
            if timeout_pct > 0:
                text += f"\nTimeout: {timeout_pct:.0f}%"

            plt.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=8)

    plt.title("Dominant Result Type with Percentages", fontsize=16)
    plt.xlabel("Number of Auctions (m)", fontsize=14)
    plt.ylabel("Number of Bidders (n)", fontsize=14)

    # Add a legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#66b3ff", label="PNE Dominant"),
        Patch(facecolor="#ff9999", label="Cycle Dominant"),
        Patch(facecolor="#ffcc99", label="Timeout Dominant"),
        Patch(facecolor="#cccccc", label="Error Dominant"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "heatmap_combined_results.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Print some statistics
    print("\nPerformance Summary:")

    # Only consider valid entries for statistics
    valid_entries = ~timeouts
    valid_mean_runtime = result_matrices["mean_runtime"][valid_entries]
    valid_mean_iterations = result_matrices["mean_iterations"][valid_entries]

    if len(valid_mean_runtime) > 0:
        print(f"Average mean execution time: {np.mean(valid_mean_runtime):.4f}s")
        print(f"Max mean execution time: {np.max(valid_mean_runtime):.4f}s")
        print(f"Min mean execution time: {np.min(valid_mean_runtime):.4f}s")
    else:
        print("No valid runtime data available.")

    if len(valid_mean_iterations) > 0:
        print(f"\nAverage mean iterations: {np.mean(valid_mean_iterations):.1f}")
        print(f"Max mean iterations: {np.max(valid_mean_iterations):.1f}")
        print(f"Min mean iterations: {np.min(valid_mean_iterations):.1f}")
    else:
        print("No valid iteration data available.")

    # Calculate overall result type percentages
    total_entries = result_matrices["pne_percentage"].size
    avg_pne_pct = np.sum(result_matrices["pne_percentage"]) / total_entries
    avg_cycle_pct = np.sum(result_matrices["cycle_percentage"]) / total_entries
    avg_timeout_pct = np.sum(result_matrices["timeout_percentage"]) / total_entries
    avg_error_pct = np.sum(result_matrices["error_percentage"]) / total_entries

    print(f"\nOverall PNE percentage: {avg_pne_pct:.1f}%")
    print(f"Overall Cycle percentage: {avg_cycle_pct:.1f}%")
    print(f"Overall Timeout percentage: {avg_timeout_pct:.1f}%")
    print(f"Overall Error percentage: {avg_error_pct:.1f}%")

    # Save raw data
    np.savez(
        os.path.join(output_dir, "heatmap_data.npz"),
        sizes=sizes,
        mean_runtime=result_matrices["mean_runtime"],
        median_runtime=result_matrices["median_runtime"],
        min_runtime=result_matrices["min_runtime"],
        max_runtime=result_matrices["max_runtime"],
        mean_iterations=result_matrices["mean_iterations"],
        median_iterations=result_matrices["median_iterations"],
        min_iterations=result_matrices["min_iterations"],
        max_iterations=result_matrices["max_iterations"],
        pne_percentage=result_matrices["pne_percentage"],
        cycle_percentage=result_matrices["cycle_percentage"],
        timeout_percentage=result_matrices["timeout_percentage"],
        error_percentage=result_matrices["error_percentage"],
        timeouts=result_matrices["timeouts"],
    )


def main():
    """Main function to run the heatmap performance test"""
    args = parse_args()

    logger.info(
        f"Starting heatmap test with ranges: {args.min_size}-{args.max_size}, step {args.step}, runs: {args.runs}"
    )

    if args.no_multiprocessing:
        logger.info("Multiprocessing is disabled. Tests will run sequentially.")
    else:
        logger.info(f"Using {args.workers} worker processes for parallel execution")

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    logger.info(f"Results will be saved to: {output_dir}")

    # Run the test
    result_matrices, all_results = run_heatmap_test(
        min_size=args.min_size,
        max_size=args.max_size,
        step=args.step,
        seed=args.seed,
        runs=args.runs,
        output_dir=output_dir,
        no_threading=args.no_threading,
        timeout=args.timeout,
        workers=args.workers,
        no_multiprocessing=args.no_multiprocessing,
    )

    # Generate final heatmaps
    create_heatmaps(
        sizes=list(range(args.min_size, args.max_size + 1, args.step)),
        result_matrices=result_matrices,
        output_dir=output_dir,
    )

    # Save detailed results
    import pickle

    with open(os.path.join(output_dir, "detailed_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    logger.info(f"Test completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
