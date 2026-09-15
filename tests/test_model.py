from src.train import load_data, train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_model_training():

    X, y = load_data(
        "data/iris.csv",
        "target"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(
        X_train,
        y_train,
        {
            "n_estimators": 100,
            "random_state": 42
        }
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    
    

def test_model_accuracy():

    X, y = load_data(
        "data/iris.csv",
        "target"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(
        X_train,
        y_train,
        {
            "n_estimators": 100,
            "random_state": 42
        }
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    assert accuracy >= 0.80