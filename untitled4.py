# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:43:58 2025

@author: user
"""

# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchmetrics import Accuracy
# from sklearn.metrics import accuracy_score  # uncomment to use sklearn

# Load preprocessed data
train_df = pd.read_csv('labelled_train.csv')
test_df = pd.read_csv('labelled_test.csv')
val_df = pd.read_csv('labelled_validation.csv')

# Separate features and labels for training, testing, and validation sets
X_train = train_df.drop('sus_label', axis=1).values
y_train = train_df['sus_label'].values
X_test = test_df.drop('sus_label', axis=1).values
y_test = test_df['sus_label'].values
X_val = val_df.drop('sus_label', axis=1).values
y_val = val_df['sus_label'].values

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
X_train = scaler.fit_transform(X_train)

# Transform the test and validation data using the fitted scaler
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Convert the numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),  # First fully connected layer
    nn.ReLU(),  # ReLU activation
    nn.Linear(128, 64),  # Second fully connected layer
    nn.ReLU(),  # ReLU activation
    nn.Linear(64, 1),  # Third fully connected layer
    nn.Sigmoid()  # Sigmoid activation for binary classification
)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss() 

optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

"""
try BCELoss() - Binary Cross Entropy Loss for binary classification, meaning the model only have 2 classes. This will provide accurate and higher probability.

try optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

Adam computes individual adaptive learning rates for different parameters from 
estimates of first and second moments of the gradients. This helps in faster convergence and better handling of sparse gradients. Adam is generally better than SGD for most problems as it adapts the learning rate for each parameter, which can lead to better performance and faster training.
"""

# Training loop
num_epoch = 10
for epoch in range(num_epoch):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients
    outputs = model(X_train_tensor)  # Forward pass: compute the model output
    loss = criterion(outputs, y_train_tensor)  # Compute the loss
    loss.backward()  # Backward pass: compute the gradients
    optimizer.step()  # Update the model parameters

# Model Evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for efficiency
    y_predict_train = model(X_train_tensor).round()  # Predict on training data and round the outputs
    y_predict_test = model(X_test_tensor).round()  # Predict on test data and round the outputs
    y_predict_val = model(X_val_tensor).round()  # Predict on validation data and round the outputs

# Calculate accuracy using torchmetrics
accuracy = Accuracy(task="binary")

train_accuracy = accuracy(y_predict_train, y_train_tensor)
test_accuracy = accuracy(y_predict_test, y_test_tensor)
val_accuracy = accuracy(y_predict_val, y_val_tensor)

# convert to int or float
train_accuracy = train_accuracy.item()
test_accuracy = test_accuracy.item()
val_accuracy = val_accuracy.item()

print("Training accuracy: {0}".format(train_accuracy))
print("Validation accuracy: {0}".format(val_accuracy))
print("Testing accuracy: {0}".format(test_accuracy))

# Calculate the accuracy using sklearn
# train_accuracy = accuracy_score(y_train_tensor, y_predict_train)
# val_accuracy = accuracy_score(y_val_tensor, y_predict_val)
# test_accuracy = accuracy_score(y_test_tensor, y_predict_test)

