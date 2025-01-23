import logging.handlers
import queue
import sys

from honours_project.simulation import Simulation
import cProfile
import pstats
import logging

logger = logging.getLogger("main")
logging.basicConfig(level=logging.DEBUG)

log_queue: queue.Queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
logger.addHandler(queue_handler)


def main() -> None:
    cool_seed = 1737125327  # 10, 7

    sim = Simulation(10, 7, shuffle=False, seed=cool_seed)
    res = sim.run()

    print(f"Seed:               {sim.seed}")
    print(f"Result:             {res}")
    print(f"Time:               {round(res.stats['time'], 2)}s")
    print(f"Time per Iteration: {1000 * round(res.stats["time"] / res.iteration, 4)}ms")


if __name__ == "__main__":
    listener = logging.handlers.QueueListener(log_queue)
    listener.start()
    try:
        main()
    finally:
        listener.stop()
