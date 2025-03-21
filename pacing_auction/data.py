from dataclasses import dataclass, field

import dataclasses
from abc import ABC

import json
from typing import Any, Iterable, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Allocation:
    """
    Allocation of a bidder to an auction
    """

    bidders: list[int]
    auction: int
    price: float  # Total price paid by all bidders


@dataclass(frozen=True, slots=True)
class BRDResult(ABC):
    """
    Base class for Best-Response Dynamics results
    """

    iteration: int
    stats: dict[str, Any] = field(kw_only=True, repr=False)


@dataclass(frozen=True, slots=True)
class Cycle(BRDResult):
    """
    Cycle state in BRD
    """

    pass


@dataclass(frozen=True, slots=True)
class PNE(BRDResult):
    """ "
    Pure Nash Equilibrium state in BRD
    """

    allocations: list[Allocation]


@dataclass(frozen=True, slots=True)
class Violation:
    """
    Violation of budgets for a bidder in an auction
    """

    bidder: int
    auction: int


@dataclass(frozen=True, slots=True)
class BestResponse:
    """
    Best response of a bidder in an auction.
    The new alpha_q is provided as well as the new utility and the old utility.
    """

    bidder: int
    new_alpha_q: int
    new_utility: float
    old_utility: float


class Distribution(Protocol):
    """
    Basic probability distribution interface
    """

    def sample(self) -> float: ...


class Uniform:
    """
    Uniform distribution

    Parameters:
        low: float
        high: float
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)


class Gaussian:
    """
    Gaussian distribution

    Parameters:
        mean: float
        std: float
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)


class Discrete:
    """
    Discrete distribution
    Chooses from a list of value with equal probability

    Parameters:
        values: Iterable[float]
    """

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
