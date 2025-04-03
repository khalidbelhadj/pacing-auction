from dataclasses import dataclass, field

import dataclasses
from abc import ABC

import json
from typing import Any

import numpy as np

from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Allocation:
    """
    Allocation of a bidder to an auction.
    Represents the outcome of a single auction allocation.

    Attributes:
        bidders: list[int]
            List of bidder IDs who won the auction
        auction: int
            ID of the auction
        price: float
            Total price paid by all bidders for this auction
    """

    __abc = 2

    bidders: list[int]
    auction: int
    price: float  # Total price paid by all bidders


@dataclass(frozen=True, slots=True)
class BRDResult(ABC):
    """
    Base class for Best-Response Dynamics results.
    Represents the outcome of a BRD process.

    Attributes:
        iteration: int
            Number of iterations the BRD process ran
        stats: dict[str, Any]
            Statistics collected during the BRD process
    """

    iteration: int
    alpha_q: NDArray[np.uint64]
    stats: dict[str, Any] = field(kw_only=True, repr=False)


@dataclass(frozen=True, slots=True)
class Cycle(BRDResult):
    """
    Cycle state in BRD.
    Represents the outcome when BRD enters a cycle of repeating states.
    """

    pass


@dataclass(frozen=True, slots=True)
class PNE(BRDResult):
    """
    Pure Nash Equilibrium state in BRD.
    Represents the outcome when BRD converges to a stable state.

    Attributes:
        x: NDArray[np.float64]
            Allocation matrix (n x m)
        p: NDArray[np.float64]
            Price vector (m)
    """

    x: NDArray[np.float64]
    p: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Violation:
    """
    Violation of budgets for a bidder in an auction.
    Represents when a bidder exceeds their budget constraint.

    Attributes:
        bidder: int
            ID of the bidder who violated their budget
        auction: int
            ID of the auction where the violation occurred
    """

    bidder: int
    auction: int


@dataclass(frozen=True, slots=True)
class BestResponse:
    """
    Best response of a bidder in an auction.
    Represents the optimal strategy adjustment for a bidder.

    Attributes:
        bidder: int
            ID of the bidder
        new_alpha_q: int
            New pacing multiplier (scaled by q)
        new_utility: float
            Utility achieved with the new pacing multiplier
        old_utility: float
            Utility with the previous pacing multiplier
    """

    bidder: int
    new_alpha_q: int
    new_utility: NDArray[np.float64]
    old_utility: NDArray[np.float64]


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder that can handle NumPy types and dataclasses.
    Extends the standard JSON encoder to properly serialize specialized types.
    """

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
