import pandas as pd

from src.train import load_data


def test_load_data():

    X, y = load_data(
        "data/iris.csv",
        "target"
    )

    assert isinstance(X, pd.DataFrame)

    assert len(X) == 150

    assert len(y) == 150

    assert X.shape[1] == 4