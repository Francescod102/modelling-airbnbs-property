# modelling-airbnbs-property
Data Science &amp; Machine Learning 
Multimodal Intelligence System

<!--  -->
This project aims to develop a framework to systematically train, tune, and evaluate a wide range of machine learning models, from simple regression to neural network models, that can be applied to various datasets.

Environment Setup

In this section, the focus was on setting up the development environment. This involved installing the necessary tools and libraries required to run the program. A virtual environment was created to keep the dependencies for the project isolated from the other projects on the machine. This step helped to avoid potential conflicts with other projects.

Data Preparation

Here, the script clean_tabular_data.py was created, which cleans a tabular dataset of Airbnb property listings. It removes or fills missing values and combines strings, returning the cleaned dataset. This script can be run as a standalone program or imported into other scripts, providing an efficient solution for data cleaning tasks. The output file clean_tabular_data.csv can be used for further analysis and machine learning applications.

Regression Models

In this part, the script modelling_regression.py was created, which loads and splits data from a CSV file and performs standardization as a preprocessing step. Then, this data is utilized to train a simple linear regression model. To assess the performance of the model, the evaluate_predictions function is defined and executed on both the training and testing data.

To optimize the model's hyperparameters, two functions have been implemented. The first function, custom_tune_regression_model_hyperparameters, allows for manual tuning of the model's hyperparameters through a nested list comprehension. The second function, tune_regression_model_hyperparameters, employs GridSearchCV to perform automated hyperparameter tuning.

Finally, a function to tune, evaluate, and save multiple regression models called evaluate_all_models is defined. It takes in a dictionary of models, along with their corresponding hyperparameters. The regression models included are linear regression, decision tree, random forest, and gradient boosting. The best overall regression model is found by the find_best_model function, which compares their metrics.

Classification Models

In this section, the modelling_classification.py script was developed, replicating the previous regression model steps, but applying them to classification models. To enhance the model's performance, the tune_classification_model_hyperparameters function was introduced, utilizing GridSearchCV for automated hyperparameter tuning.

To prevent overfitting, two techniques were implemented. The first is cross-validation, which involves splitting the dataset into training and validation sets and assessing the model's performance on the validation set. This approach can identify overfitting and allow for model adjustments. The second technique is regularization, which adds a penalty term to the model's loss function to discourage overfitting.

Neural Network Model

A neural network model is defined in modelling_nn.py script and it is trained using the specified dataset and configuration. The performance of the model was evaluated by running it 15 times with different parametrizations. After that, the best model was selected, and its characteristics and metrics are reported below.

The dataset is created by the NightlyPriceRegressionDataset class, which loads the data file and returns tuples of features and labels. The data is split into training, validation, and testing sets, and the data loaders shuffle the data and batch it for efficient processing during training.

The neural network model is defined by the NeuralNetwork class, which takes as input a configuration dictionary, the number of input features, and the number of output features. The configuration dictionary specifies the number of hidden layers, the width of each hidden layer, and the learning rate and optimizer used for training.

Specifically, the generate_nn_configs function creates a list of configuration dictionaries by randomly selecting values for the number of hidden layers, the width of each hidden layer, the learning rate, and the optimizer. These configurations are used to train a separate instance of the neural network model.

The train function initializes an optimizer and a TensorBoard writer, and then loops through the specified number of epochs, measuring the start time and inference latencies for each batch. At the end of each epoch, it computes the metrics for the training and validation sets. The selected model has the following hyperparameters and metrics:

Parameters	Value		Metrics	Value
Optimizer	Adagrad		RMSE Loss	55.909
Learning Rate	0.1		R-squared	0.525
Hidden Layer Width	7		Training Duration	5.608''
Hidden Layer Depth	3		Inference Latency	0.003''
Framework Test

To further test the generalizability of the neural network, the same process has been repeated with a different label to verify that the model works similarly on any dataset. The results have shown that the neural network can indeed be applied to other datasets, achieving similar performance with different labels. This indicates that the model is robust and can be used for a wide range of business problems.