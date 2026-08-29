from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_model.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first with: python src/train.py"
        )

    model = joblib.load(MODEL_PATH)

    sample = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3],
    ])

    predictions = model.predict(sample)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
