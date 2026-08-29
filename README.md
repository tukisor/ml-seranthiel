# lil ml project

this project is a minimal machine learning starter using python

## Setup

```bash
python -m venv .venv
. .venv/bin/activate   # macOS/Linux
# or .\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Train the model

```bash
python src/train.py
```

## Run a sample prediction

```bash
python src/predict.py
```

this trains a simple classifier on the iris dataset and saves the model to the `models` folder.
