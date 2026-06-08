import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(split: str = "train"):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class AdImageDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"])

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        content_label = torch.tensor(int(row["content_type_label"]), dtype=torch.long)
        mood_label = torch.tensor(int(row["mood_label"]), dtype=torch.long)

        return {"image": image, "content_type": content_label, "mood": mood_label}
