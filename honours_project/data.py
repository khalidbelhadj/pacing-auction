from dataclasses import dataclass
import json
from typing import NamedTuple


@dataclass(frozen=True)
class Allocation:
    bidder: int
    auction: int
    price: float

    def __repr__(self) -> str:
        return f"Allocation(bidder={self.bidder}, auction={self.auction}, price={self.price})"


@dataclass(frozen=True)
class SimulationResult:
    time: float
    iteration: int


@dataclass(frozen=True)
class Cycle(SimulationResult):
    pass


@dataclass(frozen=True)
class PNE(SimulationResult):
    allocations: list[Allocation]


@dataclass(frozen=True)
class FPAResult:
    pass


@dataclass(frozen=True)
class Violation(FPAResult):
    bidder: int
    auction: int


@dataclass(frozen=True)
class FPAAllocation(FPAResult):
    allocations: list[Allocation]


@dataclass(frozen=True)
class BestResponse:
    new_alpha: float
    new_utility: float
    old_utility: float
