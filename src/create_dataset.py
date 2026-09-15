from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path


def create_dataset():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    output_path = Path("data/iris.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    create_dataset()