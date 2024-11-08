from attr import dataclass


@dataclass
class State:
    n: int
    m: int
    q: int
    budget: list[float]
    valuation: list[list[float]]
    alpha: list[float]
