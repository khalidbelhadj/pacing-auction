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
from dataclasses import dataclass
import os
import sys
import logging
import time
from datetime import datetime
import traceback
import json
import numpy as np
import pandas as pd
from typing import Any, NamedTuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from pacing_auction import elimination
from pacing_auction.auction import Auction
from pacing_auction.data import Cycle
from pacing_auction.generator import (
    BinaryAuctionGenerator,
    CompleteAuctionGenerator,
    SampledAuctionGenerator,
    CorrelatedAuctionGenerator,
)

logger = logging.getLogger("results")


@dataclass
class Job:
    id: int
    n: int
    m: int
    run: int
    no_threading: bool
    timeout: int
    generator: str
    elim_strategy: str
    sigma: float | None
    delta: float | None
    seed: int | None


@dataclass
class ResultData:
    id: int = -1
    n: int = -1
    m: int = -1
    generator: str = "complete"
    elim_strategy: str = "subsequent"
    sigma: float | None = None
    delta: float | None = None
    runtime: float = -1
    iterations: int = -1
    result_type: str = "Error"
    timeout: bool = False
    error: bool = False
    cycle_length: int = 0
    v_matrix: np.ndarray | None = None
    b_vector: np.ndarray | None = None
    alpha_q_vector: np.ndarray | None = None
    utility_stats: list | None = None
    alpha_q_stats: list | None = None
    social_welfare_stats: list | None = None
    liquid_welfare_stats: list | None = None


def parse_args():
    """Parse command line arguments, handling config file and CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive auction test suite. Loads config from --config file if provided, CLI args override config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults
    )

    # Config file argument (parsed first)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON configuration file. CLI arguments will override values from this file.",
    )

    parser.add_argument(
        "--continue",
        dest="continue_dir",
        type=str,
        help="Path to a previous run directory to continue execution from",
    )

    # Size parameters
    parser.add_argument("--min-n", type=int, help="Minimum number of bidders")
    parser.add_argument("--max-n", type=int, help="Maximum number of bidders")
    parser.add_argument("--step-n", type=int, help="Step size for n values")
    parser.add_argument("--min-m", type=int, help="Minimum number of auctions")
    parser.add_argument("--max-m", type=int, help="Maximum number of auctions")
    parser.add_argument("--step-m", type=int, help="Step size for m values")

    # Generator parameters
    parser.add_argument(
        "--generators",
        type=str,
        nargs="+",
        help="list of generators (complete/correlated/sampled) to test",
    )
    parser.add_argument(
        "--elim-strategies",
        type=str,
        nargs="+",
        help="list of elimination strategies (all/current/subsequent) to test",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        help="list of sigma values for correlated generator",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        help="list of delta values for correlated generator",
    )

    # Test parameters
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        const=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of runs for each configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results, relative to project root",
    )
    parser.add_argument(
        "--no-threading",
        action=argparse.BooleanOptionalAction,  # Use BooleanOptionalAction for clearer --no-threading/--threading
        help="Disable threading in the auction algorithm",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum time (in seconds) for a single configuration (0 for no timeout)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes for parallel execution",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        help="Number of tests completed before saving intermediate results",
    )

    # --- Default values ---
    # Set defaults here, they might be overridden by config file, then by CLI args
    default_args = {
        "min_n": 2,
        "max_n": 10,
        "step_n": 1,
        "min_m": 2,
        "max_m": 10,
        "step_m": 1,
        "generators": ["complete", "correlated", "sampled", "binary"],
        "elim_strategies": ["subsequent", "all"],
        "sigmas": [0.05, 0.1, 0.2, 0.3],
        "deltas": [0.0, 0.25, 0.5, 0.75],
        "seed": None,
        "runs": 5,
        "output_dir": "results",
        "no_threading": False,  # Default to threading enabled
        "timeout": 0,
        "workers": max(1, multiprocessing.cpu_count() - 1),
        "save_interval": 100,
    }
    parser.set_defaults(**default_args)

    # --- Load config file ---
    # Parse only the --config argument first
    temp_args, _ = parser.parse_known_args()
    config_values = {}
    if temp_args.config and os.path.exists(temp_args.config):
        try:
            with open(temp_args.config, "r") as f:
                config_values = json.load(f)
            # Override defaults with config file values
            parser.set_defaults(**config_values)
            logger.info(f"Loaded configuration from {temp_args.config}")
        except Exception as e:
            logger.error(f"Error loading config file {temp_args.config}: {e}")
            sys.exit(1)
    elif temp_args.config:
        logger.error(f"Config file not found: {temp_args.config}")
        sys.exit(1)

    # --- Parse all arguments ---
    # CLI arguments will now override defaults and config file values
    args = parser.parse_args()

    # Make sure threading flag is boolean
    args.no_threading = bool(args.no_threading)

    return args


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


def run_single_test(job: Job) -> ResultData:
    """Run a single test for a given configuration"""
    c = job
    result_data = ResultData()

    result_data.id = c.id
    result_data.n = c.n
    result_data.m = c.m
    result_data.generator = c.generator
    result_data.elim_strategy = c.elim_strategy
    result_data.sigma = c.sigma
    result_data.delta = c.delta

    # Record start time
    start_time = time.perf_counter()

    try:
        # Select generator based on type
        if c.generator == "complete":
            generator = CompleteAuctionGenerator()
        elif c.generator == "correlated":
            if c.sigma is None or c.delta is None:
                raise ValueError(
                    "Sigma and delta must be provided for correlated generator"
                )
            generator = CorrelatedAuctionGenerator(sigma=c.sigma, delta=c.delta)
        elif c.generator == "sampled":
            generator = SampledAuctionGenerator()
        elif c.generator == "binary":
            generator = BinaryAuctionGenerator()
        else:
            raise ValueError(f"Unknown generator type: {c.generator}")

        elim_strategy: elimination.ElimStrategy = elimination.Subsequent
        if c.elim_strategy == "current":
            elim_strategy = elimination.Current
        elif c.elim_strategy == "all":
            elim_strategy = elimination.All
        elif c.elim_strategy == "subsequent":
            elim_strategy = elimination.Subsequent
        else:
            print(
                f"Unknown elimination strategy: {c.elim_strategy}, defaulting to Subsequent"
            )

        auction = Auction(
            n=c.n,
            m=c.m,
            rng=np.random.default_rng(c.seed),
            collect_stats=True,
            threaded=not c.no_threading,
            generator=generator,
            elim=elim_strategy,
            no_budget=c.generator == "binary",
        )

        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Auction run timed out after {c.timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(c.timeout)

        result = auction.responses()
        signal.alarm(0)

        # Record end time
        exec_time = time.perf_counter() - start_time

        # Store results
        result_data.runtime = exec_time
        result_data.iterations = result.iteration
        result_data.result_type = type(result).__name__

        # Store the large matrices/vectors
        result_data.v_matrix = auction._v
        result_data.b_vector = auction._b
        result_data.alpha_q_vector = auction._alpha_q

        # Store the new list-based stats
        result_data.utility_stats = result.stats.get("utility")
        result_data.alpha_q_stats = result.stats.get("alpha_q")
        result_data.social_welfare_stats = result.stats.get("social_welfare")
        result_data.liquid_welfare_stats = result.stats.get("liquid_welfare")

        if isinstance(result, Cycle):
            result_data.cycle_length = result.stats.get("cycle_length", 0)

    except TimeoutError:
        logger.warning(
            f"Configuration ({c.n}, {c.m}) timed out after {c.timeout} seconds"
        )
        result_data.runtime = c.timeout
        result_data.timeout = True
        result_data.result_type = "Timeout"

    except Exception as e:
        logger.error(f"Error running configuration ({c.n}, {c.m}): {str(e)}")
        logger.error(traceback.format_exc())
        result_data.error = True

    return result_data


def save_all_results(all_results: list[ResultData], output_dir: str):
    csv_results = [
        {
            k: v
            for k, v in r.__dict__.items()
            if k
            not in [
                "v_matrix",
                "b_vector",
                "alpha_q_vector",
                "utility_stats",
                "alpha_q_stats",
                "social_welfare_stats",
                "liquid_welfare_stats",
            ]
        }
        for r in all_results
    ]
    csv_df = pd.DataFrame(csv_results)
    csv_df.to_csv(
        os.path.join(output_dir, "data.csv"),
        index=False,
    )

    # Save matrices and vectors separately (convert to list for saving if needed)
    np.savez_compressed(
        os.path.join(output_dir, "auction_matrices.npz"),
        v_matrices=np.array(
            [
                (result.v_matrix.tolist() if result.v_matrix is not None else None)
                for result in all_results
            ],
            dtype=object,
        ),
        b_vectors=np.array(
            [
                (result.b_vector.tolist() if result.b_vector is not None else None)
                for result in all_results
            ],
            dtype=object,
        ),
        alpha_q_vectors=np.array(
            [
                (
                    result.alpha_q_vector.tolist()
                    if result.alpha_q_vector is not None
                    else None
                )
                for result in all_results
            ],
            dtype=object,
        ),
        indices=np.array(list(range(len(all_results)))),
    )

    # Save new list stats separately
    np.savez_compressed(
        os.path.join(output_dir, "auction_stats.npz"),
        utility=np.array(
            [result.utility_stats for result in all_results], dtype=object
        ),
        alpha_q_stats=np.array(
            [result.alpha_q_stats for result in all_results],
            dtype=object,
        ),
        social_welfare_stats=np.array(
            [result.social_welfare_stats for result in all_results],
            dtype=object,
        ),
        liquid_welfare_stats=np.array(
            [result.liquid_welfare_stats for result in all_results],
            dtype=object,
        ),
        indices=np.array(list(range(len(all_results)))),
    )


def run_test_suite(
    jobs: list[Job],
    output_dir: str,
    workers: int,
    save_interval: int,
) -> None:
    """Run the comprehensive test suite"""
    total_jobs = len(jobs)
    completed = 0

    start_time = time.time()
    all_results = []

    # Process configurations in parallel using ProcessPoolExecutor
    logger.info(f"Total tests to run: {total_jobs}")
    logger.info(f"Running tests with {workers} worker processes")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all configurations to the executor
        future_to_job = {
            executor.submit(run_single_test, config): config for config in jobs
        }

        # Process results as they complete
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                all_results.append(result)

                completed += 1
                logger.info(
                    f"Completed job {job.id} ({job.n}, {job.m}) with {job.generator} generator and {job.elim_strategy} elimination strategy"
                    + (
                        f" (sigma={job.sigma}, delta={job.delta})"
                        if job.sigma is not None
                        else ""
                    )
                    + f" - Progress: {completed}/{total_jobs} tests ({completed/total_jobs*100:.1f}%)"
                )

                # Save intermediate results if needed
                if completed % save_interval == 0:
                    save_all_results(all_results, output_dir)
                    logger.info(f"Saved intermediate results after {completed} tests")

            except Exception as e:
                logger.error(f"Error processing configuration ({job.n}, {job.m}):")
                logger.error(traceback.format_exc())

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    save_all_results(all_results, output_dir)

    # Print summary statistics
    print("\nTest Suite Summary:")
    print(f"Total tests run: {total_jobs}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")


def generate_jobs(
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
    no_threading: bool,
    timeout: int,
    runs: int,
):
    # Generate the ranges of sizes
    n_values = list(range(min_n, max_n + 1, step_n))
    m_values = list(range(min_m, max_m + 1, step_m))

    # Create a list of all configurations to test
    configs: list[Job] = []
    id = 0
    for n in n_values:
        for m in m_values:
            for run in range(runs):
                for generator in generators:
                    for elim_strategy in elim_strategies:
                        if generator == "correlated":
                            for sigma in sigmas:
                                for delta in deltas:
                                    configs.append(
                                        Job(
                                            id=id,
                                            n=n,
                                            m=m,
                                            run=run,
                                            no_threading=no_threading,
                                            timeout=timeout,
                                            generator=generator,
                                            elim_strategy=elim_strategy,
                                            sigma=sigma,
                                            delta=delta,
                                            seed=seed,
                                        )
                                    )
                                    id += 1
                        else:
                            configs.append(
                                Job(
                                    id=id,
                                    n=n,
                                    m=m,
                                    run=run,
                                    no_threading=no_threading,
                                    timeout=timeout,
                                    generator=generator,
                                    elim_strategy=elim_strategy,
                                    sigma=None,
                                    delta=None,
                                    seed=seed,
                                )
                            )
                            id += 1
    return configs


def find_missing_jobs(continue_dir: str) -> list[Job]:
    """Find configurations that haven't been completed yet from a previous run"""
    # Path to the config file
    config_path = os.path.join(continue_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {continue_dir}")

    # Load the configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Path to the data file
    data_path = os.path.join(continue_dir, "data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found in {continue_dir}")

    # Load the data that has already been processed
    completed_df = pd.read_csv(data_path)

    # Find the maximum ID used in the existing data
    max_id = completed_df["id"].max() if not completed_df.empty else -1

    # Create a dictionary to count successful completions for each configuration
    # The key is (n, m, generator, elim_strategy, sigma, delta)
    completion_counts = {}

    # Count successful completions
    for _, row in completed_df.iterrows():
        # Skip configurations that resulted in an error or timeout
        if (not pd.isna(row["error"]) and row["error"]) or (
            not pd.isna(row["timeout"]) and row["timeout"]
        ):
            continue

        # For configurations with sigma/delta, use the actual values
        sigma = row["sigma"] if not pd.isna(row["sigma"]) else None
        delta = row["delta"] if not pd.isna(row["delta"]) else None

        # Create a key for this configuration
        config_key = (
            int(row["n"]),
            int(row["m"]),
            row["generator"],
            row["elim_strategy"],
            sigma,
            delta,
        )

        # Increment the count for this configuration
        if config_key in completion_counts:
            completion_counts[config_key] += 1
        else:
            completion_counts[config_key] = 1

    # Generate missing configurations
    missing_jobs = []
    next_id = max_id + 1

    # Generate the ranges of sizes
    n_values = list(range(config["min_n"], config["max_n"] + 1, config["step_n"]))
    m_values = list(range(config["min_m"], config["max_m"] + 1, config["step_m"]))

    # For each unique configuration
    for n in n_values:
        for m in m_values:
            for generator in config["generators"]:
                for elim_strategy in config["elim_strategies"]:
                    if generator == "correlated":
                        for sigma in config["sigmas"]:
                            for delta in config["deltas"]:
                                # Create a key for this configuration
                                cfg_key = (n, m, generator, elim_strategy, sigma, delta)

                                # Get how many successful runs we have
                                completed_count = completion_counts.get(cfg_key, 0)

                                # Calculate how many more runs we need
                                missing_runs = config["runs"] - completed_count

                                # Add the missing runs
                                for run in range(missing_runs):
                                    missing_jobs.append(
                                        Job(
                                            id=next_id,
                                            n=n,
                                            m=m,
                                            run=run,
                                            no_threading=config["no_threading"],
                                            timeout=config["timeout"],
                                            generator=generator,
                                            elim_strategy=elim_strategy,
                                            sigma=sigma,
                                            delta=delta,
                                            seed=config["seed"],
                                        )
                                    )
                                    next_id += 1
                    else:
                        # Create a key for this configuration
                        cfg_key = (n, m, generator, elim_strategy, None, None)

                        # Get how many successful runs we have
                        completed_count = completion_counts.get(cfg_key, 0)

                        # Calculate how many more runs we need
                        missing_runs = config["runs"] - completed_count

                        # Add the missing runs
                        for run in range(missing_runs):
                            missing_jobs.append(
                                Job(
                                    id=next_id,
                                    n=n,
                                    m=m,
                                    run=run,
                                    no_threading=config["no_threading"],
                                    timeout=config["timeout"],
                                    generator=generator,
                                    elim_strategy=elim_strategy,
                                    sigma=None,
                                    delta=None,
                                    seed=config["seed"],
                                )
                            )
                            next_id += 1

    logger.info(
        f"Found {len(missing_jobs)} missing configurations out of {len(n_values) * len(m_values) * len(config['generators']) * len(config['elim_strategies']) * config['runs']} total configurations"
    )
    return missing_jobs


# Configure logging
def setup_logging(log_file: str | None = None):
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Add file handler
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    return logging.getLogger("results")


def main():
    """Main function to run the test suite"""
    global logger
    args = parse_args()

    log_file = None

    # Determine if we're continuing from a previous run
    if args.continue_dir:
        configs = find_missing_jobs(args.continue_dir)
        output_dir = args.continue_dir  # Use the same output directory
        log_file = os.path.join(output_dir, log_file) if log_file else None
        logger = setup_logging(log_file)
        logger.info(f"Logging to file: {log_file}")
        logger.info(f"Continuing from previous run in directory: {args.continue_dir}")
    else:
        # Generate new configurations
        configs = generate_jobs(
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
            no_threading=args.no_threading,
            timeout=args.timeout,
        )

        # Create output directory
        output_dir = create_output_dir(args.output_dir)

        # Update logger to also log to file
        log_file = os.path.join(output_dir, log_file) if log_file else None
        logger = setup_logging(log_file)
        logger.info(f"Logging to file: {log_file}")

        logger.info(f"Results will be saved to: {output_dir}")
        # Save the final configuration used for this run
        config_to_save = vars(args)
        config_to_save.pop("config", None)
        config_to_save.pop("continue_dir", None)  # Remove the continue_dir parameter
        config_save_path = os.path.join(output_dir, "config.json")
        try:
            with open(config_save_path, "w") as f:
                json.dump(config_to_save, f, indent=4)
            logger.info(f"Saved final configuration to {config_save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_save_path}: {e}")

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

    # Run the test suite
    run_test_suite(
        jobs=configs,
        output_dir=output_dir,
        workers=args.workers,
        save_interval=args.save_interval,
    )

    logger.info(f"Test suite completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
