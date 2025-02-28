from concurrent.futures import ProcessPoolExecutor, as_completed
import logging.handlers
import threading
import time

import pandas as pd

from honours_project.data import AuctionResult
from honours_project.auction import Auction
import logging

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)


def print_result(sim: Auction, res: AuctionResult):
    print(f"Seed:               {sim.seed}")
    print(f"Result:             {type(res)}, {res.iteration}")
    print(f"Time:               {round(res.stats['time'], 2)}s")
    print(f"Time per Iteration: {1000 * round(res.stats["time"] / res.iteration, 4)}ms")


def run_sim(n, m, number_of_simulations):
    result = []
    for _ in range(number_of_simulations):
        sim = Auction(n, m, collect_stats=True)
        res = sim.run()
        print_result(sim, res)
        result.append((sim.n, sim.m, res))
    return result


def collect():
    ns = [5]
    ms = range(10, 20 + 1)
    number_of_simulations = 1

    start_time = time.time()

    results_lock = threading.Lock()

    futures = []
    executor = ProcessPoolExecutor()

    for n in ns:
        for m in ms:
            futures.append(executor.submit(run_sim, n, m, number_of_simulations))

    for future in as_completed(futures):
        with results_lock:
            res = future.result()

            first = res[0]
            logger.info(
                f"Finished {first[0]}, {first[1]} in {time.time() - start_time:.2f}s"
            )

            data = [(n, m, r.iteration) for n, m, r in res]
            df = pd.DataFrame(data, columns=["n", "m", "result"])

            file_name = f"data/results-{time.strftime('%Y-%m-%d-%H-%M')}.csv"
            df.to_csv(file_name, mode="a", header=False, index=False)


def main() -> None:
    print("started")
    sim = Auction(
        10,
        10,
        threaded=True,
        no_budget=True,
        seed=741714086,
        cache_utility=True,
    )
    res = sim.run()
    print_result(sim, res)
    print(sim.utility_cache_hits / (sim.utility_cache_hits + sim.utility_cache_misses))


if __name__ == "__main__":
    main()
