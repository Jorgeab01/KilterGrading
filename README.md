# KilterGrading

![Status](https://img.shields.io/badge/status-in%20progress-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.13-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Predicts the difficulty grade of a Kilter Board climbing problem from its
hold layout and wall angle, trained on real climbs graded by the community.

## Why

Grading a climbing problem is subjective. Different climbers disagree
depending on height, strength style, and experience. On a Kilter Board, the
displayed grade is actually the average of every climber who's logged that
problem, which means a newly set route has no grade until people have climbed
it and voted.

KilterGrading predicts a starting grade from the hold layout and wall angle, 
before any community feedback exists, using gradient boosting trained
on 190k already graded climbs.

## How

Each climb's hold layout and wall angle are parsed into a set of features, 
then used with the real community grade to train an XGBoost model that learns 
the relationship between hold layout and difficulty.

## Data

The database isn't included in this repo.

Download `kilter_splits.sqlite` from Hugging Face and place it in `data/`:

https://huggingface.co/datasets/Vilin97/KilterBoard

Running `src/data.py` creates the feature datasets:

```text
data/train_features.csv
data/val_features.csv
data/test_features.csv
```

## Usage

```bash
pip install -r requirements.txt
python src/data.py
python src/train_model.py 1
```

`train_model.py` takes the iteration number to run, defined in
`src/features.py`. It trains a new model if one doesn't already exist
for that number, or loads the saved one and just evaluates it.

## Results

Current validation MAE: **2.07**.

Last iteration uses features that describe the shape of the route, not 
just counts: biggest reach between holds, how far the route spreads 
sideways and up, and hold-type proportions. 

See [ITERATIONS.md](ITERATIONS.md) for the details and results of each
iteration.

## Roadmap

- [x] `src/data.py`: reads the database, converts each route into a feature
    vector, and saves the train, validation, and test feature datasets.

- [x] `src/train_model.py`: trains an XGBoost model, evaluates it on the
    validation set, and saves the trained model.

- [ ] `src/predict.py`: loads a trained model and predicts a grade for a
    new route.

- [ ] Web interface: a clickable board where anyone can place holds and
    get a live grade prediction.

- [ ] Sequence model (stretch goal): represent routes as an ordered
    sequence of moves and compare an LSTM against the XGBoost baseline.