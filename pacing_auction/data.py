from dataclasses import dataclass, field

import dataclasses
from abc import ABC

import json
from typing import Any, Iterable, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Allocation:
    bidders: list[int]
    auction: int
    price: float  # Total price paid by all bidders


@dataclass(frozen=True, slots=True)
class AuctionResult(ABC):
    iteration: int
    stats: dict[str, Any] = field(kw_only=True, repr=False)


@dataclass(frozen=True, slots=True)
class Cycle(AuctionResult):
    pass


@dataclass(frozen=True, slots=True)
class PNE(AuctionResult):
    allocations: list[Allocation]


@dataclass(frozen=True, slots=True)
class Violation:
    bidder: int
    auction: int


@dataclass(frozen=True, slots=True)
class FPAAllocation:
    allocations: list[Allocation]
    utility: float


@dataclass(frozen=True, slots=True)
class BestResponse:
    bidder: int
    new_alpha_q: int
    new_utility: float
    old_utility: float


class Distribution(Protocol):
    def sample(self) -> float: ...


class Uniform:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)


class Gaussian:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)


class Discrete:
    def __init__(self, values: Iterable[float]):
        self.values = values

    def sample(self) -> float:
        return np.random.choice(list(self.values))


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        # Dataclass serialisation
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore

        # Numpy serialisation
        if isinstance(o, (np.bool)):
            return bool(o)  # type: ignore
        elif isinstance(o, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):  # type: ignore
            return int(o)  # type: ignore
        elif isinstance(o, (np.uint8, np.uint16, np.uint32, np.uint64)):  # type: ignore
            return int(o)  # type: ignore
        elif isinstance(o, (np.float16, np.float32, np.float64)):  # type: ignore
            return float(o)  # type: ignore
        elif isinstance(o, (np.complex64, np.complex128)):  # type: ignore
            return {"real": o.real, "imag": o.imag}  # type: ignore
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        elif isinstance(o, (np.bool)):
            return bool(o)  # type: ignore
        elif isinstance(o, (np.void)):
            return None

        return super().default(o)
