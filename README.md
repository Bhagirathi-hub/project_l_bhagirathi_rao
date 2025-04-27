# project_l_bhagirathi_rao
Project Proposal: Binary Classification of Serengeti Zebras and Elephants Using Image Processing 

Input-Output Statement 
• Input: Camera trap image of a zebra or elephant in .jpg format. 
• Output: 
o Classification label: 
▪ 0: Zebra 
▪ 1: Elephant 

Data Source 
The project will use the "Snapshot Serengeti" dataset available on Kaggle.  
Link: https://www.kaggle.com/datasets/gauravduttakiit/animal-classification-challenge 

For this project, the image classification will be performed using the VGG11 architecture. 

NOTE : In the data folder I have 2 subfolders as train_zebras and train_elephants
I have trained 7000 random images from these folders spiltted as 3500 each for zebras and elephants

Zebra-Elephant-Classification/
│
├── data/                          
│   ├── train_zebra/               
│   └── train_elephant/            
│
├── checkpoints/                   
│   └── final_weights.pth          
│
├── model.py                       
├── train.py                       
├── dataset.py                    
├── config.py                     
├── predict.py                     
├── README.md                      


# Zebra and Elephant Classification Model

## Overview

This project involves training a deep learning model to classify images of zebras and elephants. The model uses a **VGG11** architecture for image classification. The dataset consists of images from the directories `train_zebra` and `train_elephant`, and the model is trained to distinguish between the two classes.

## Requirements

- Python 3.6 or higher
- PyTorch
- torchvision
- PIL
- numpy

You can install the required libraries using the following command:
pip install -r requirements.txt
Where requirements.txt includes:

torch==<specific_version>
torchvision==<specific_version>
Pillow==<specific_version>
numpy==<specific_version>
Dataset
The dataset consists of images of zebras and elephants located in the following directories:

data/train_zebra/: Contains images of zebras.

data/train_elephant/: Contains images of elephants.

Each category contains 3500 images.

The images are resized to a specified resolution (configured in the config.py file).

Model Architecture
The model is based on the VGG11 architecture, which is a deep Convolutional Neural Network (CNN). It consists of 11 layers and is commonly used for image classification tasks. The architecture includes:

Convolutional Layers: To extract features from the images.

Max Pooling Layers: To downsample the feature maps.

Fully Connected Layers: To classify the images based on the extracted features.

Softmax Activation: To output probabilities for each class (Zebra/Elephant).

Training
Training Script
The training script train.py trains the model on the dataset and saves the trained model weights to a specified checkpoint directory.

python train.py
Parameters
num_epochs: 5 (Number of epochs for training)

batch_size: 64 (Batch size for the data loader)

learning_rate: 0.001 (Learning rate for the optimizer)

checkpoint_dir: Directory to save model weights (default: /content/drive/My Drive/project_l_bhagirathi_rao/checkpoints/final_weights.pth)

python train.py --num_epochs 5 --batch_size 64 --learning_rate 0.001
This will train the model for 5 epochs, using a batch size of 64 and a learning rate of 0.001.

Evaluation
The model can be evaluated on a separate validation or test dataset. You can modify the code to load the saved weights and perform predictions.

To load and evaluate the model:

import torch
from model import UnicornNet  # or your model file
model = UnicornNet()
model.load_state_dict(torch.load("path_to_trained_weights.pth"))
model.eval()

# Example inference (modify this to match your data pipeline)
image = ...  # Load image
output = model(image)
Save and Load Model
Saving Model
The model is saved after training to the specified checkpoint directory:

torch.save(model.state_dict(), "/path_to_checkpoint/final_weights.pth")
Loading Model
To load the model weights for inference or further training:

model = UnicornNet()  # Define your model architecture
model.load_state_dict(torch.load("/path_to_checkpoint/final_weights.pth"))
model.eval()

