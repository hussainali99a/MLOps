from src.data_validation import validate_data


def test_valid_dataset():

    result = validate_data(
        "data/iris.csv"
    )

    assert result is True