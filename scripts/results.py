#!/usr/bin/env python3
"""
Script to generate comprehensive test data for auctions with different parameters and generators.
This script will test all combinations of n and m values, different auction generators,
and for the correlated generator, different sigma values.

The script stores the following data for each test:
- Standard metrics: runtime, iterations, result type (PNE/Cycle), etc.
- Welfare metrics: social welfare, liquid welfare, and revenue
- Auction data: valuation matrix (v), budget vector (b), and pacing multiplier vector (alpha_q)

The matrices and vectors are stored separately in a compressed NPZ file for efficiency.
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime
import traceback
import numpy as np
import pandas as pd
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from pacing_auction import elimination
from pacing_auction.auction import Auction
from pacing_auction.data import Cycle
from pacing_auction.generator import (
    CompleteAuctionGenerator,
    SampledAuctionGenerator,
    CorrelatedAuctionGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("test_suite")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run comprehensive auction test suite")

    # Size parameters
    parser.add_argument(
        "--min-n", type=int, default=2, help="Minimum number of bidders (default: 2)"
    )
    parser.add_argument(
        "--max-n", type=int, default=10, help="Maximum number of bidders (default: 10)"
    )
    parser.add_argument(
        "--step-n", type=int, default=1, help="Step size for n values (default: 1)"
    )
    parser.add_argument(
        "--min-m", type=int, default=2, help="Minimum number of auctions (default: 2)"
    )
    parser.add_argument(
        "--max-m", type=int, default=10, help="Maximum number of auctions (default: 10)"
    )
    parser.add_argument(
        "--step-m", type=int, default=1, help="Step size for m values (default: 1)"
    )

    # Generator parameters
    parser.add_argument(
        "--generators",
        type=str,
        nargs="+",
        default=["complete", "correlated", "sampled"],
        help="list of generators (complete/correlated/sampled) to test (default: complete correlated sampled)",
    )
    parser.add_argument(
        "--elim-strategies",
        type=str,
        nargs="+",
        default=["subsequent"],
        help="list of elimination strategies (all/current/subsequent) to test (default: subsequent)",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3],
        help="list of sigma values for correlated generator (default: 0.05 0.1 0.2 0.3)",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
        help="list of delta values for correlated generator (default: 0.0 0.25 0.5 0.75)",
    )

    # Test parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=time.time_ns() % 2**32,
        help="Random seed for reproducibility (default: current time in nanoseconds)",
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
        default="results",
        help="Directory to save results, relative to project root (default: 'results')",
    )
    parser.add_argument(
        "--no-threading",
        action="store_true",
        help="Disable threading in the auction algorithm",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Maximum time (in seconds) to allow for a single configuration (default: 0 - no timeout)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help=f"Number of worker processes for parallel execution (default: {max(1, multiprocessing.cpu_count() - 1)} - autodetected)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Number of tests completed before saving intermediate results (default: 100)",
    )

    return parser.parse_args()


def create_output_dir(output_dir: str) -> str:
    """Create output directory with timestamp"""
    # Make the output directory relative to project_root
    output_dir = os.path.join(project_root, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_dir = os.path.join(output_dir, f"{timestamp}")
    os.makedirs(run_dir)

    return run_dir


def run_single_test(
    n: int,
    m: int,
    seed: int,
    no_threading: bool,
    timeout: int,
    generator_type: str,
    elim_strategy_type: str,
    sigma: float | None = None,
    delta: float | None = None,
) -> dict[str, Any]:
    """Run a single test for a given configuration"""
    result_data = {
        "runtime": -1,
        "iterations": -1,
        "result_type": "Error",
        "timeout": False,
        "error": False,
        "cycle_length": 0,
        "max_social_welfare": 0,
        "max_liquid_welfare": 0,
        "max_revenue": 0,
        "min_social_welfare": 0,
        "min_liquid_welfare": 0,
        "min_revenue": 0,
        "v_matrix": None,
        "b_vector": None,
        "alpha_q_vector": None,
    }

    # Record start time
    start_time = time.perf_counter()

    try:
        # Select generator based on type
        if generator_type == "complete":
            generator = CompleteAuctionGenerator()
        elif generator_type == "correlated":
            if sigma is None or delta is None:
                raise ValueError(
                    "Sigma and delta must be provided for correlated generator"
                )
            generator = CorrelatedAuctionGenerator(sigma=sigma, delta=delta)
        else:  # sampled
            generator = SampledAuctionGenerator()

        elim_strategy: elimination.ElimStrategy = elimination.Subsequent
        if elim_strategy_type == "current":
            elim_strategy = elimination.Current
        elif elim_strategy_type == "all":
            elim_strategy = elimination.All
        elif elim_strategy_type == "subsequent":
            elim_strategy = elimination.Subsequent
        else:
            print(
                f"Unknown elimination strategy: {elim_strategy_type}, defaulting to Subsequent"
            )

        auction = Auction(
            n=n,
            m=m,
            seed=seed,
            collect_stats=True,
            threaded=not no_threading,
            generator=generator,
            elim=elim_strategy,
        )

        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Auction run timed out after {timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        result = auction.responses()
        signal.alarm(0)

        # Record end time
        exec_time = time.perf_counter() - start_time

        # Store results
        result_data["runtime"] = exec_time
        result_data["iterations"] = result.iteration
        result_data["result_type"] = type(result).__name__

        result_data["max_social_welfare"] = result.stats["max_social_welfare"]
        result_data["max_liquid_welfare"] = result.stats["max_liquid_welfare"]
        result_data["max_revenue"] = result.stats["max_revenue"]
        result_data["min_social_welfare"] = result.stats["min_social_welfare"]
        result_data["min_liquid_welfare"] = result.stats["min_liquid_welfare"]
        result_data["min_revenue"] = result.stats["min_revenue"]

        # Store the additional data requested: v matrix, b vector, and alpha_q
        result_data["v_matrix"] = (
            auction._v.tolist()
        )  # Convert numpy array to list for JSON serialization
        result_data["b_vector"] = auction._b.tolist()
        result_data["alpha_q_vector"] = auction._alpha_q.tolist()

        if isinstance(result, Cycle):
            result_data["cycle_length"] = result.stats.get("cycle_length", 0)

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
    config: tuple[
        int, int, int, int, bool, int, int, str, str, float | None, float | None, int
    ],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Process a single configuration with multiple runs"""
    (
        i,
        j,
        n,
        m,
        no_threading,
        timeout,
        runs,
        generator_type,
        elim_strategy,
        sigma,
        delta,
        base_seed,
    ) = config

    config_results = []

    for run_idx in range(runs):
        run_seed = base_seed + run_idx
        result = run_single_test(
            n,
            m,
            run_seed,
            no_threading,
            timeout,
            generator_type,
            elim_strategy,
            sigma,
            delta,
        )
        config_results.append(result)

    return (i, j, config_results)


def run_test_suite(
    min_n: int,
    max_n: int,
    step_n: int,
    min_m: int,
    max_m: int,
    step_m: int,
    generators: list[str],
    elim_strategies: list[str],
    sigmas: list[float],
    deltas: list[float],
    seed: int | None,
    runs: int,
    output_dir: str,
    no_threading: bool,
    timeout: int,
    workers: int,
    save_interval: int,
) -> None:
    """Run the comprehensive test suite"""
    # Generate the ranges of sizes
    n_values = list(range(min_n, max_n + 1, step_n))
    m_values = list(range(min_m, max_m + 1, step_m))

    # Create a list of all configurations to test
    configs = []
    for i, n in enumerate(n_values):
        for j, m in enumerate(m_values):
            for generator_type in generators:
                for elim_strategy in elim_strategies:
                    if generator_type == "correlated":
                        # For correlated generator, test each sigma and delta value combination
                        for sigma in sigmas:
                            for delta in deltas:
                                configs.append(
                                    (
                                        i,
                                        j,
                                        n,
                                        m,
                                        no_threading,
                                        timeout,
                                        runs,
                                        generator_type,
                                        elim_strategy,
                                        sigma,
                                        delta,
                                        seed,
                                    )
                                )
                    else:
                        # For other generators, no sigma or delta needed
                        configs.append(
                            (
                                i,
                                j,
                                n,
                                m,
                                no_threading,
                                timeout,
                                runs,
                                generator_type,
                                elim_strategy,
                                None,
                                None,
                                seed,
                            )
                        )

    total_configs = len(configs)
    total_tests = total_configs * runs
    completed = 0

    start_time = time.time()

    # Store all results
    all_results = []
    df = pd.DataFrame()

    # Process configurations in parallel using ProcessPoolExecutor
    logger.info(f"Running tests with {workers} worker processes")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all configurations to the executor
        future_to_config = {
            executor.submit(process_configuration, config): config for config in configs
        }

        # Process results as they complete
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            i, j, n, m, generator_type, elim_strategy, sigma, delta = (
                config[0],
                config[1],
                config[2],
                config[3],
                config[7],
                config[8],
                config[9],
                config[10],
            )

            try:
                i, j, config_results = future.result()

                # Store results
                for result in config_results:
                    result.update(
                        {
                            "n": n,
                            "m": m,
                            "generator": generator_type,
                            "elim_strategy": elim_strategy,
                            "sigma": sigma,
                            "delta": delta,
                        }
                    )
                    all_results.append(result)

                completed += runs
                logger.info(
                    f"Completed configuration ({n}, {m}) with {generator_type} generator and {elim_strategy} elimination strategy"
                    + (f" (sigma={sigma}, delta={delta})" if sigma is not None else "")
                    + f" - Progress: {completed}/{total_tests} tests ({completed/total_tests*100:.1f}%)"
                )

                # Save intermediate results if needed
                if completed % save_interval == 0:
                    df = pd.DataFrame(all_results)
                    # Remove the large matrix/vector fields from the CSV for efficiency
                    csv_results = [
                        {
                            k: v
                            for k, v in r.items()
                            if k not in ["v_matrix", "b_vector", "alpha_q_vector"]
                        }
                        for r in all_results
                    ]
                    csv_df = pd.DataFrame(csv_results)
                    csv_df.to_csv(
                        os.path.join(output_dir, "data.csv"),
                        index=False,
                    )

                    # Save matrices and vectors separately
                    np.savez_compressed(
                        os.path.join(output_dir, "auction_matrices.npz"),
                        v_matrices=np.array(
                            [
                                result["v_matrix"]
                                for result in all_results
                                if result["v_matrix"] is not None
                            ],
                            dtype=object,
                        ),
                        b_vectors=np.array(
                            [
                                result["b_vector"]
                                for result in all_results
                                if result["b_vector"] is not None
                            ],
                            dtype=object,
                        ),
                        alpha_q_vectors=np.array(
                            [
                                result["alpha_q_vector"]
                                for result in all_results
                                if result["alpha_q_vector"] is not None
                            ],
                            dtype=object,
                        ),
                        indices=list(range(len(all_results))),
                    )
                    logger.info(f"Saved intermediate results after {completed} tests")

            except Exception as e:
                logger.error(f"Error processing configuration ({n}, {m}): {str(e)}")
                logger.error(traceback.format_exc())

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Save final results
    # Remove large matrices from CSV for efficiency
    csv_results = [
        {
            k: v
            for k, v in r.items()
            if k not in ["v_matrix", "b_vector", "alpha_q_vector"]
        }
        for r in all_results
    ]
    csv_df = pd.DataFrame(csv_results)
    csv_df.to_csv(os.path.join(output_dir, "data.csv"), index=False)

    # Save all matrices and vectors to a separate file in NPZ format for more efficient storage
    np.savez_compressed(
        os.path.join(output_dir, "auction_matrices.npz"),
        v_matrices=np.array(
            [
                result["v_matrix"]
                for result in all_results
                if result["v_matrix"] is not None
            ],
            dtype=object,
        ),
        b_vectors=np.array(
            [
                result["b_vector"]
                for result in all_results
                if result["b_vector"] is not None
            ],
            dtype=object,
        ),
        alpha_q_vectors=np.array(
            [
                result["alpha_q_vector"]
                for result in all_results
                if result["alpha_q_vector"] is not None
            ],
            dtype=object,
        ),
        indices=list(range(len(all_results))),
    )

    # Print summary statistics
    print("\nTest Suite Summary:")
    print(f"Total configurations tested: {total_configs}")
    print(f"Total tests run: {total_tests}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"\nResults by generator:")
    for generator_type in generators:
        generator_results = csv_df[csv_df["generator"] == generator_type]
        if len(generator_results) > 0:
            print(f"\n{generator_type.title()} Generator:")
            print(
                f"  PNE percentage: {(generator_results['result_type'] == 'PNE').mean()*100:.1f}%"
            )
            print(
                f"  Cycle percentage: {(generator_results['result_type'] == 'Cycle').mean()*100:.1f}%"
            )
            print(
                f"  Timeout percentage: {generator_results['timeout'].mean()*100:.1f}%"
            )
            print(f"  Error percentage: {generator_results['error'].mean()*100:.1f}%")
            print(f"  Average runtime: {generator_results['runtime'].mean():.2f}s")
            print(f"  Average iterations: {generator_results['iterations'].mean():.1f}")

    if "correlated" in generators:
        print("\nResults by sigma and delta for correlated generator:")
        for sigma in sigmas:
            for delta in deltas:
                sigma_delta_results = csv_df[
                    (csv_df["generator"] == "correlated")
                    & (csv_df["sigma"] == sigma)
                    & (csv_df["delta"] == delta)
                ]
                if len(sigma_delta_results) > 0:
                    print(f"\nSigma = {sigma}, Delta = {delta}:")
                    print(
                        f"  PNE percentage: {(sigma_delta_results['result_type'] == 'PNE').mean()*100:.1f}%"
                    )
                    print(
                        f"  Cycle percentage: {(sigma_delta_results['result_type'] == 'Cycle').mean()*100:.1f}%"
                    )
                    print(
                        f"  Timeout percentage: {sigma_delta_results['timeout'].mean()*100:.1f}%"
                    )
                    print(
                        f"  Error percentage: {sigma_delta_results['error'].mean()*100:.1f}%"
                    )
                    print(
                        f"  Average runtime: {sigma_delta_results['runtime'].mean():.2f}s"
                    )
                    print(
                        f"  Average iterations: {sigma_delta_results['iterations'].mean():.1f}"
                    )


def main():
    """Main function to run the test suite"""
    args = parse_args()

    logger.info(
        f"Starting test suite with ranges: n={args.min_n}-{args.max_n} (step {args.step_n}), "
        f"m={args.min_m}-{args.max_m} (step {args.step_m}), "
        f"generators={args.generators}, "
        f"elim_strategies={args.elim_strategies}, "
        f"sigmas={args.sigmas}, "
        f"deltas={args.deltas}, "
        f"runs={args.runs}"
    )

    logger.info(f"Using {args.workers} worker processes for parallel execution")

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    logger.info(f"Results will be saved to: {output_dir}")

    # Run the test suite
    run_test_suite(
        min_n=args.min_n,
        max_n=args.max_n,
        step_n=args.step_n,
        min_m=args.min_m,
        max_m=args.max_m,
        step_m=args.step_m,
        generators=args.generators,
        elim_strategies=args.elim_strategies,
        sigmas=args.sigmas,
        deltas=args.deltas,
        seed=args.seed,
        runs=args.runs,
        output_dir=output_dir,
        no_threading=args.no_threading,
        timeout=args.timeout,
        workers=args.workers,
        save_interval=args.save_interval,
    )

    logger.info(f"Test suite completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
