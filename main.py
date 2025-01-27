import logging.handlers
import queue

from honours_project.data import BestResponse, SimulationResult
from honours_project.simulation import Simulation
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

log_queue: queue.Queue[int] = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
logger.addHandler(queue_handler)


def print_result(sim: Simulation, res: SimulationResult):
    print(f"Seed:               {sim.seed}")
    print(f"Result:             {type(res)}, {res.iteration}")
    print(f"Time:               {round(res.stats['time'], 2)}s")
    print(f"Time per Iteration: {1000 * round(res.stats["time"] / res.iteration, 4)}ms")


def main() -> None:
    sim = Simulation(5, 5, collect_stats=True)

    cont = False

    def on_best_response(s: Simulation, res: BestResponse):
        nonlocal cont
        logger.info(f"Best Response: {res}")
        if cont:
            return

        command = input()
        if command == "c":
            cont = True

    res = sim.run(on_best_response)
    print_result(sim, res)


if __name__ == "__main__":
    listener = logging.handlers.QueueListener(log_queue)
    listener.start()
    try:
        main()
    finally:
        listener.stop()
