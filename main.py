from elimination import Subsequent
from simulation import Simulation


def main() -> None:
    sim = Simulation(elimination=Subsequent())
    allocations = sim.run()
    for a in allocations:
        print(a)


if __name__ == "__main__":
    main()
