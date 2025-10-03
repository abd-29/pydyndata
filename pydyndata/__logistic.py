import pandas as pd
from pandas import DataFrame


def make_logistic(
    r: float = 0.1,
    K: int = 100,
    N0: int = 10,
    n_steps: int = 100
) -> DataFrame:
    """
    Generate a dataset based on the discrete logistic equation:
    N_{t+1} = N_t + r * N_t * (1 - N_t / K)

    Parameters
    ----------
    r : float
        Growth rate.
    K : int
        Carrying capacity.
    N0 : int
        Initial population.
    n_steps : int
        Number of iterations.

    Returns
    -------
    DataFrame
        A dataset with two columns:
        - "t": time step
        - "N": population size
    """
    values: list[float] = [N0]
    for _ in range(1, n_steps):
        Nt: float = values[-1] + r * values[-1] * (1 - values[-1] / K)
        values.append(Nt)

    df: DataFrame = pd.DataFrame({"t": range(n_steps), "N": values})
    return df
