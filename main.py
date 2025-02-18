from concurrent.futures import ProcessPoolExecutor, as_completed
import logging.handlers
import threading
import time

import pandas as pd

from honours_project.data import SimulationResult
from honours_project.simulation import Simulation
import logging

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)


def print_result(sim: Simulation, res: SimulationResult):
    print(f"Seed:               {sim.seed}")
    print(f"Result:             {type(res)}, {res.iteration}")
    print(f"Time:               {round(res.stats['time'], 2)}s")
    print(f"Time per Iteration: {1000 * round(res.stats["time"] / res.iteration, 4)}ms")


def run_sim(n, m, number_of_simulations):
    result = []
    for _ in range(number_of_simulations):
        sim = Simulation(n, m, collect_stats=True)
        res = sim.run()
        result.append((sim.n, sim.m, res))
    return result


def collect():
    ns = [10]
    ms = range(1, 21)
    number_of_simulations = 10

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
    collect()


if __name__ == "__main__":
    main()
