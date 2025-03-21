from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class ElimStrategy(Protocol):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.int8]) -> None: ...


class Subsequent(ElimStrategy):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.int8]) -> None:
        mask[bidder, auction:] = 0


class All(ElimStrategy):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.int8]) -> None:
        mask[bidder, :] = 0


class Current(ElimStrategy):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.int8]) -> None:
        mask[bidder, auction] = 0
