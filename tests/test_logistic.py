from pydyndata import make_logistic
import pandas as pd


def test_make_logistic_returns_dataframe() -> None:
    """Check that make_logistic returns a non-empty DataFrame with expected columns."""
    df = make_logistic(r=0.1, K=100, N0=10, n_steps=20)

    # Ensure type is DataFrame
    assert isinstance(df, pd.DataFrame)

    # Ensure DataFrame is not empty
    assert not df.empty

    # Ensure DataFrame has correct columns
    assert set(df.columns) == {"t", "N"}
