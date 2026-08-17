import sys
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from features import ITERATIONS

def get_xy(data_type, features):

    """
    Loads the given feature CSV type and splits it into input and target

    Args: 
        data_type: "train", "val" or "test"
        features: list of features to use as model inputs

    Returns:
        (X, y) tuple: Inputs and outputs for the model
        
    Raises:
        ValueError: if data_type is not one of the three valid options.

    """

    if data_type not in ("test", "train", "val"):
        raise ValueError("Invalid parameter. Valid: test, train, val")

    data = pd.read_csv(f"./data/{data_type}_features.csv")

    X = data[features]
    y = data["grade"]

    return X, y

def train_model(X_train, y_train, model_params):

    """
    Trains an XGBoost model on the given data.

    Args:
        X_train: features.
        y_train: targets with the real grades.
        model_params: dict of parameters passed to XGBRegressor.

    Returns:
        Trained model.
    """

    model = XGBRegressor(**model_params)

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_val, y_val):

    """
    Evaluates a trained model against the given val set.

    Args:
        model: a trained XGBRegressor.
        X_val: features to predict on.
        y_val: real grades to compare predictions against.

    Returns:
        Mean absolute error between predictions and the real grades.
    """

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)

    return mae


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Usage: python train_model.py <iteration_num>")

    iteration = int(sys.argv[1])
    
    if iteration not in ITERATIONS:
        raise ValueError(f"Iteration {iteration} does not exist")

    iteration_data = ITERATIONS[iteration]

    features = iteration_data["features"]
    model_params = iteration_data["model_params"]  

    X_train, y_train = get_xy("train", features)
    X_val, y_val = get_xy("val", features)

    model_path = f"./models/it{iteration:03d}.json"

    if os.path.exists(model_path):

        print(f"Loading existing model: {model_path}")

        model = XGBRegressor()
        model.load_model(model_path)
    else:

        print(f"Training model: {model_path}")

        model = train_model(X_train, y_train, model_params)
        model.save_model(model_path)
    
    mae = evaluate_model(model, X_val, y_val)
    print(f"Validation MAE: {mae:.2f}")

