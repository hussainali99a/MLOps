import pandas as pd


REQUIRED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target"
]


def validate_data(data_path):

    df = pd.read_csv(data_path)

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty")

    # Check missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError(
            "Dataset contains missing values"
        )

    # Check target values
    valid_targets = {0, 1, 2}

    actual_targets = set(
        df["target"].unique()
    )

    if not actual_targets.issubset(valid_targets):
        raise ValueError(
            f"Invalid target values: {actual_targets}"
        )

    return True