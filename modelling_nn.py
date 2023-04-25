# Import libraries
import datetime
import numpy as np
import os
import json
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torchmetrics import R2Score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Import methods defined previously
from tabular_data import load_data

# Create a PyTorch dataset that returns a tuple when indexed
class NightlyPriceRegressionDataset(Dataset):

    """Initialize the dataset and load the input"""
    def __init__(self):
        super().__init__()
        data = pd.read_csv("../data/clean_tabular_data.csv", index_col = "ID")
        self.X, self.y = load_data(data, "Price_Night")
        self.X = self.X.select_dtypes(include = ["int64", "float64"])

    """Retrieve a single data point from the dataset given an index"""
    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index]).float()
        label = torch.tensor(self.y.iloc[index]).float()
        return (features, label)

    """Return the total number of data points in the dataset"""
    def __len__(self):
        return len(self.X)

# Create a function that returns a dictionary containing data loaders for each sets
def get_data_loaders(dataset, batch_size):

    """Split the data into training, validation, and test sets"""
    train_data, test_data = torch.utils.data.random_split(dataset, lengths = [0.8, 0.2])
    train_data, validation_data = torch.utils.data.random_split(train_data, lengths = [0.7, 0.3])

    """Create data loaders for each set that shuffles the data"""
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = True)

    """Add the results to a dictionary"""
    dataloader = {
        "Train": train_loader,
        "Validation": validation_loader,
        "Testing": test_loader
    }

    return dataloader

# Define a function which creates many configuration dictionaries for your network
def generate_nn_configs():

    """Define the lists of possible values"""
    optimiser_list = ["Adagrad", "Adam", "SGD"]
    learning_rate_list = [0.1, 0.01, 0.001]
    width_list = [6, 7, 8]
    depth_list = [3, 4, 5]

    """Generate configuration dictionaries"""
    nn_configs = []
    for i in range(15):
        optimiser = random.choice(optimiser_list)
        learning_rate = random.choice(learning_rate_list)
        hidden_layer_width = random.choice(width_list)
        hidden_layer_depth = random.choice(depth_list)

        config = {
            "optimiser": optimiser,
            "learning_rate": learning_rate,
            "hidden_layer_width": hidden_layer_width,
            "hidden_layer_depth": hidden_layer_depth
        }

        nn_configs.append(config)

    """Write a configuration file"""
    with open('nn_config.yaml', 'w') as file:
        yaml.dump(nn_configs, file, sort_keys = False, default_flow_style = False)

    return nn_configs

# Define a function which reads the configuration file
def get_nn_config():
    with open("nn_config.yaml", "r") as stream:
        try:
            nn_config = yaml.safe_load(stream)
            return nn_config
        except yaml.YAMLError as e:
            print(e)

# Create "Neural Network" model in Pytorch
class NeuralNetwork(nn.Module):

    """Initialize the hyperparameters of the network"""
    def __init__(self, config, in_features, out_features):
        super().__init__()
        ## Extract hyperparameters from the configuration file
        hidden_layer_depth = config["hidden_layer_depth"]
        hidden_layer_width = config["hidden_layer_width"]
        ## Create a list to store the layers of the network
        layers = []
        ## Loop over the specified number of hidden layers
        for i in range(hidden_layer_depth):
            ## Create a linear layer with the specified input and output dimensions
            ## The input dimension is the size of the input layer if this is the first hidden layer
            ## Otherwise, it is the width of the previous hidden layer
            ## The output dimension is the width of the hidden layer
            layers.append(nn.Linear(in_features if i == 0 else hidden_layer_width, hidden_layer_width))
            ## Add a ReLU activation function after each hidden layer
            layers.append(nn.ReLU())
        ## Add the output layer with a linear activation function to the list of layers
        layers.append(nn.Linear(hidden_layer_width, out_features))
        ## Create the network as a sequential module using the list of layers
        self.layers = nn.Sequential(*layers)

    """Define the forward pass of the network"""
    def forward(self, input):
        return self.layers(input)

# Train a PyTorch model using the specified dataset and parameters
def train(model, dataloader, config, epochs = 10):

    """Define the optimiser and TensorBoard writer"""
    optimiser_class = getattr(torch.optim, config["optimiser"])
    optimiser = optimiser_class(model.parameters(), lr = config["learning_rate"])
    writer = SummaryWriter()

    """Initialize the batch index"""
    batch_idx = 0
    batch_idx2 = 0

    """Measure the start time of training"""
    inference_latencies = []
    start_time = time.time()

    """Loop through the specified number of epochs"""
    for epoch in range(epochs):
        
        """Loop through each batch in the DataLoader"""
        for batch in dataloader["Train"]:
            ## Extract the features and labels from the batch
            features, labels = batch
            labels = torch.unsqueeze(labels, 1)
            ## Measure the start time of inference
            inference_start_time = time.time()
            ## Compute the model's prediction and the corresponding loss
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            ## Compute performance metrics
            rmse_train = torch.sqrt(loss)
            r2_train = R2Score()
            r2_train = r2_train(prediction, labels)
            ## Measure the end time of inference
            inference_end_time = time.time()
            ## Compute the inference latency and add it to a list
            inference_latency = inference_end_time - inference_start_time
            inference_latencies.append(inference_latency)
            ## Perform an optimization step and zero the gradients
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            ## Visualise the loss function using TensorBoard
            writer.add_scalars("Train Loss", {"loss": loss.item()}, batch_idx)
            ## Increment the batch index
            batch_idx += 1

        """Loop through each batch in the DataLoader"""
        for batch in dataloader["Validation"]:
            ## Extract the features and labels from the batch
            features, labels = batch
            labels = torch.unsqueeze(labels, 1)
            ## Compute the model's prediction and the corresponding loss
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            ## Compute performance metrics
            rmse_validation = torch.sqrt(loss)
            r2_validation = R2Score()
            r2_validation = r2_validation(prediction, labels)
            ## Visualise the loss function using TensorBoard
            writer.add_scalars("Validation Loss", {"loss": loss.item()}, batch_idx2)
            ## Increment the batch index
            batch_idx2 += 1

    """Compute the training duration and average inference latency"""
    end_time = time.time()
    training_duration = round(end_time - start_time, 3)
    avg_inference_latency = round(sum(inference_latencies) / len(inference_latencies), 3)

    """Store the performance metrics in a dictionary"""
    metrics = {
        "rmse_loss": {"train": round(rmse_train.item(), 3), "validation": round(rmse_validation.item(), 3)},
        "r_squared": {"train": round(r2_train.item(), 3), "validation": round(r2_validation.item(), 3)},  
        "training_duration": training_duration,
        "inference_latency": avg_inference_latency
        }

    return metrics

# Define a function to save the model
def save_model(model, hyperparams, metrics):

    """Check if the object is a PyTorch module"""
    if isinstance(model, nn.Module):

        """Create the folder with the current date and time as its name"""
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("models", "neural_networks", "regression", now)
        os.makedirs(path, exist_ok = True)
        
        """Save the model, its hyperparameters, and its metrics"""
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "hyperparameters.json"), "w") as f:
            json.dump(hyperparams, f)
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f)

# Define a function to find the best overall neural network configuration
def find_best_nn(dataloaders):

    best_params = {}
    best_rmse = float("inf")
    best_metrics = {}

    """Generate a list of configurations"""
    configs = generate_nn_configs()

    for config in configs:

        """Create a new model using the configuration"""
        model = NeuralNetwork(config, in_features = 11, out_features = 1)

        """Train the model and get the evaluation metrics"""
        metrics = train(model, dataloaders, config)

        """Compare metrics to find the best model"""
        if metrics["rmse_loss"]["validation"] < best_rmse:
            best_rmse = metrics["rmse_loss"]["validation"] # test
            best_model = model
            best_params = config
            best_metrics = metrics

    """Return the best model, its configuration, and evaluation metrics"""
    return best_model, best_params, best_metrics

# Ensure that the code inside it is only executed if the script is being run directly
if __name__ == "__main__":
    np.random.seed(1)
    dataset = NightlyPriceRegressionDataset()
    dataloaders = get_data_loaders(dataset, batch_size = 12)
    get_nn_config()
    best_model, best_params, best_metrics = find_best_nn(dataloaders)
    print(f"Model: {type(best_model)}, Hyperparameters: {best_params}, Metrics: {best_metrics}")
    save_model(best_model, best_params, best_metrics)