# type: ignore
from dataclasses import dataclass, field

import dataclasses
from abc import ABC

import json
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Allocation:
    bidders: list[int]
    auction: int
    price: float  # Total price paid by all bidders


@dataclass(frozen=True, slots=True)
class SimulationResult(ABC):
    time: float
    iteration: int
    stats: dict[str, Any] = field(default_factory=dict, kw_only=True, repr=False)


@dataclass(frozen=True, slots=True)
class Cycle(SimulationResult):
    pass


@dataclass(frozen=True, slots=True)
class PNE(SimulationResult):
    allocations: list[Allocation]


@dataclass(frozen=True, slots=True)
class FPAResult(ABC):
    pass


@dataclass(frozen=True, slots=True)
class Violation(FPAResult):
    bidder: int
    auction: int


@dataclass(frozen=True, slots=True)
class FPAAllocation(FPAResult):
    allocations: list[Allocation]


@dataclass(frozen=True, slots=True)
class BestResponse:
    new_alpha_q: float
    new_utility: float
    old_utility: float


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        # Dataclass serialisation
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        # Numpy serialisation
        if isinstance(o, (np.bool_)):
            return bool(o)
        elif isinstance(
            o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
        ):
            return int(o)
        elif isinstance(o, (np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(o)
        elif isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.complex64, np.complex128)):
            return {"real": o.real, "imag": o.imag}
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        elif isinstance(o, (np.bool)):
            return bool(o)
        elif isinstance(o, (np.void)):
            return None

        return super().default(o)
