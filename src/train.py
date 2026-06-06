import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from src.image_model import AdImageModel
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

from src.dataset import AdImageDataset

NUM_CONTENT = 3
NUM_MOODS = 4


def get_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
