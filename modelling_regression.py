# Import libraries
import joblib
import json
import numpy as np
import os
import pandas as pd
import warnings

from sklearn import ensemble, linear_model, metrics, model_selection, preprocessing, tree

# Import methods defined previously
from tabular_data import load_data

# Define the file path and load the data
df = pd.read_csv("../data/clean_tabular_data.csv", index_col = "ID")
X, y = load_data(df, "Price_Night")

# Split the data into training, validation, and test sets
X = X.select_dtypes(include = ["int64", "float64"])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5)

# Print the shape of training data
print(X_train.shape, y_train.shape)

# Fit and transform the data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

# Create a dictionary containing the training, validation, and testing data
datasets = {
    "X_train": X_train,
    "y_train": y_train, 
    "X_test": X_test, 
    "y_test": y_test, 
    "X_validation": X_validation, 
    "y_validation": y_validation
}

# Create a simple regression model to predict the nightly cost of each listing
# sgdr = linear_model.SGDRegressor()
# sgdr.fit(X_train, y_train)

# Make predictions on the training and testing data
# y_train_pred = sgdr.predict(X_train)
# y_test_pred = sgdr.predict(X_test)

# Define a function to evaluate the predictions on the training and testing data
# def evaluate_model_performance(y_train, y_train_pred, y_test, y_test_pred):

#     """Create a dictionary to store the results"""
#     results = {}
    
#     """Evaluate the model performance on the training set"""
#     mse_train = round(metrics.mean_squared_error(y_train, y_train_pred), 3)
#     rmse_train = round(np.sqrt(mse_train), 3)
#     r2_train = round(metrics.r2_score(y_train, y_train_pred), 3)

#     """Add the training results to the dictionary"""
#     results["train"] = {
#         "mse": mse_train,
#         "rmse": rmse_train,
#         "r_squared": r2_train
#     }

#     """Evaluate the model performance on the testing set"""
#     mse_test = round(metrics.mean_squared_error(y_test, y_test_pred), 3)
#     rmse_test = round(np.sqrt(mse_test), 3)
#     r2_test = round(metrics.r2_score(y_test, y_test_pred), 3)

#     """Add the testing results to the dictionary"""
#     results["test"] = {
#         "mse": mse_test,
#         "rmse": rmse_test,
#         "r_squared": r2_test
#     }
    
#     return results

# Implement a custom function to tune the hyperparameters of the model
def custom_tune_regression_model_hyperparameters(model_class, datasets, hyperparameters):

    """Create a nested list comprehensions of hyperparameter combinations"""
    combinations = [{"alpha": alpha, "learning_rate": learning_rate, "max_iter": max_iter, "tol": tol, "penalty": penalty}
                    for alpha in hyperparameters["alpha"]
                    for learning_rate in hyperparameters["learning_rate"]
                    for max_iter in hyperparameters["max_iter"]
                    for tol in hyperparameters["tol"]
                    for penalty in hyperparameters["penalty"]]

    """Find the best model based on validation set"""
    best_model = None
    best_params = {}
    best_rmse = float("inf")
    best_metrics = {}

    for combination in combinations:
        model = model_class(alpha = combination["alpha"], learning_rate = combination["learning_rate"], max_iter = combination["max_iter"], tol = combination["tol"], penalty = combination["penalty"])
        model.fit(datasets["X_train"], datasets["y_train"])

        """Use the trained model to make predictions on the validation set"""
        y_pred_validation = model.predict(datasets["X_validation"])
        rmse_validation = round(metrics.mean_squared_error(datasets["y_validation"], y_pred_validation, squared = False), 3)
        r2_validation = round(metrics.r2_score(datasets["y_validation"], y_pred_validation), 3)

        """
        The combination of hyperparameters that results in the lowest 
        RMSE on the validation set is selected as the best hyperparameters
        """
        if rmse_validation < best_rmse:
            best_model = model
            best_params = combination
            best_rmse = rmse_validation
            best_metrics["validation_rmse"] = rmse_validation
            best_metrics["validationr_r_squared"] = r2_validation

    """Return the best model, its hyperparameters, and performance metrics"""
    return best_model, best_params, best_metrics

# Use "GridSearchCV" to tune the hyperparameters of the model
def tune_regression_model_hyperparameters(model_class, datasets, hyperparameters):

    """Tuning hyperparameters"""
    grid = model_selection.GridSearchCV(model_class, hyperparameters) # verbose = 10
    grid.fit(datasets["X_train"], datasets["y_train"])

    """Model prediction on validation set using best hyperparameters"""
    y_val_pred = grid.predict(datasets["X_validation"])

    """Calculate metrics on validation set"""
    rmse_validation = round(metrics.mean_squared_error(datasets["y_validation"], y_val_pred, squared = False), 3)
    r2_validation = round(metrics.r2_score(datasets["y_validation"], y_val_pred), 3)

    """Store best model, best hyperparameters, and best metrics in a dictionary"""
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_metrics = {"rmse": rmse_validation, "r_squared": r2_validation}
    
    return best_model, best_params, best_metrics

# Define a function to save the model
def save_model(path, best_model, best_params, best_metrics):

    """Create the folder if it doesn't exist"""
    os.makedirs(path, exist_ok = True)
    
    """Save the model, its hyperparameters, and its metrics"""
    joblib.dump(best_model, os.path.join(path, "model.joblib"))
    with open(os.path.join(path, "hyperparameters.json"), "w") as f:
        json.dump(best_params, f)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(best_metrics, f)

# Create a dictionary containing the ranges of the hyperparameters to be tuned
sgdr_hyperparams = {
    "loss": ["huber", "epsilon_insensitive"],
    "alpha": [0.0001, 0.001],
    "learning_rate": ["optimal", "adaptive"],
    "max_iter": [2000, 3000],
    "penalty": ["l1", "l2"]
}
decision_tree_hyperparams = {
    "criterion": ["friedman_mse", "absolute_error"],
    "splitter": ["best", "random"],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
random_forest_hyperparams = {
    "n_estimators": [50, 100],
    "criterion": ["friedman_mse", "absolute_error"],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
gradient_boosting_hyperparams = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# Define a dictionary of models with corresponding hyperparameters and folder names
models = {
    "linear_regression": {
        "model_class": linear_model.SGDRegressor(),
        "hyperparameters": sgdr_hyperparams
    },
    "decision_tree": {
        "model_class": tree.DecisionTreeRegressor(),
        "hyperparameters": decision_tree_hyperparams
    },
    "random_forest": {
        "model_class": ensemble.RandomForestRegressor(),
        "hyperparameters": random_forest_hyperparams
    },
    "gradient_boosting": {
        "model_class": ensemble.GradientBoostingRegressor(),
        "hyperparameters": gradient_boosting_hyperparams
    }
}

# Define a function that tune, evaluate, and save all the models
def evaluate_all_models(models):

    for model_name, model_info in models.items():

        model_class = model_info["model_class"]
        hyperparameters = model_info["hyperparameters"]

        best_model, best_params, best_metrics = tune_regression_model_hyperparameters(model_class, datasets, hyperparameters)
        print(f"Model: {best_model}, RMSE: {best_metrics['rmse']:.3f}, R-Squared: {best_metrics['r_squared']:.3f}")

        path = os.path.join("models", "regression", model_name)
        save_model(path, best_model, best_params, best_metrics)
    
# Define a function to find the best overall regression model
def find_best_model(models):

    best_model = None
    best_params = {}
    best_rmse = float("inf")
    best_metrics = {}

    for model_name, model_info in models.items():

        """Load model class from the dictionary"""
        model_class = model_info["model_class"]

        """Load hyperparameters and metrics from JSON files"""
        hyperparams_path = os.path.join("models", "regression", model_name, "hyperparameters.json")
        with open(hyperparams_path, "r") as f:
            hyperparameters = json.load(f)
        metrics_path = os.path.join("models", "regression", model_name, "metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        """Compare metrics to find the best model"""
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]

            best_model = model_class
            best_params = hyperparameters
            best_metrics = metrics

    return best_model, best_params, best_metrics

# Ensure that the code inside it is only executed if the script is being run directly
if __name__ == "__main__":
    np.random.seed(1)
    warnings.filterwarnings("ignore")
    evaluate_all_models(models)
    best_model, best_params, best_metrics = find_best_model(models)
    print(f"Model: {best_model}, Hyperparameters: {best_params}, RMSE: {best_metrics['rmse']:.3f}, R-Squared: {best_metrics['r_squared']:.3f}")