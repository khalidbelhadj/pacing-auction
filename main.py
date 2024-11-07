from elimination import Subsequent
from simulation import Simulation


def main() -> None:
    sim = Simulation(file="state.json", elimination=Subsequent(), check_cycle=True)
    allocations = sim.run()
    for a in allocations:
        print(a)


if __name__ == "__main__":
    main()
