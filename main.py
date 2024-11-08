from elimination import All
from simulation import Simulation


def main() -> None:
    sim = Simulation(elimination=All())
    allocations = sim.run()
    for a in allocations:
        print(a)


if __name__ == "__main__":
    main()
