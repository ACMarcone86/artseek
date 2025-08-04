from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


class AquaDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", transform=None):
        """Initialize the Aqua dataset.

        Args:
            data_dir (str | Path): Path to dataset directory
            split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to "train".
            transform (_type_, optional): Transform to apply to the dataset. Defaults to None.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load annotations
        with open(self.data_dir / "annotations" / f"{split}.json", "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = self.data_dir / "images" / ann["image"]
        image = Image.open(img_path).convert("RGB")
        ann["image"] = image

        if self.transform:
            ann = self.transform(ann)

        return ann
