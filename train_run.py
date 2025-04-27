import torch
import torch.nn as nn
import torch.optim as optim
from dataset import unicornLoader
from model import UnicornNet
from train import train_model
from config import batch_size, num_epochs, learning_rate

# Load data
train_loader = unicornLoader("data", batch_size=batch_size)


# Initialize model
model = UnicornNet(num_classes=2)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, num_epochs=num_epochs, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
