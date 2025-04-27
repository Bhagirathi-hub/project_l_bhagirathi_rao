# interface.py

# replace MyCustomModel with the name of your model
from model import UnicornNet as TheModel

# replace my_descriptively_named_train_function with your training function
from train import train_model as the_trainer

# replace cryptic_inf_f with your predictor function
from predict import classify_animals as the_predictor

# replace UnicornImgDataset and unicornLoader with your dataset and dataloader
from dataset import UnicornImgDataset as TheDataset
from dataset import unicornLoader as the_dataloader

# import batch size and epochs
from config import batch_size as the_batch_size
from config import epochs as total_epochs
