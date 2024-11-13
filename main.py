from simulation import Simulation
import cProfile
import pstats


def main() -> None:
    sim = Simulation(n=10, m=10, q=100)
    allocations = sim.run()
    for a in allocations:
        print(a)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()

    # main()
