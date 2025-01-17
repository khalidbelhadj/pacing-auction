from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class ElimStrategy(Protocol):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.bool]) -> None: ...


class Subsequent(ElimStrategy):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.bool]) -> None:
        mask[bidder, auction:] = False


class All(ElimStrategy):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.bool]) -> None:
        mask[bidder, :] = False


class Current(ElimStrategy):
    @staticmethod
    def eliminate(bidder: int, auction: int, mask: NDArray[np.bool]) -> None:
        mask[bidder, auction] = False
