import os

import torch

dataset_path = os.getenv("DATASET_PATH")  # path to where the dataset is saved

seed = 42

preprocess_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logs_path = "results/logs"
os.makedirs(logs_path, exist_ok=True)

models_path = "results/models"
os.makedirs(models_path, exist_ok=True)
