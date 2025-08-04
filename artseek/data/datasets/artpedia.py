import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class ArtpediaDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", transform=None):
        """Initialize the dataset from the paper "Explain me the painting".

        Args:
            data_dir (str | Path): Path to dataset directory
            split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to "train".
            transform (_type_, optional): Transform to apply to the dataset. Defaults to None.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load annotations
        with open(self.data_dir / "artpedia.json", "r") as f:
            self.annotations = json.load(f)

        # make annotations become a list of dictionaries
        self.new_annotations = []
        for k, v in tqdm(self.annotations.items(), desc="Processing annotations"):
            v["id"] = k
            # check if the image exists
            img_path = self.data_dir / "images" / f"{k}.jpg"
            if img_path.exists():
                v["img"] = f"{k}.jpg"
                self.new_annotations.append(v)

        # replace annotations with the new list
        self.annotations = self.new_annotations

        # keep only annotations of the specified split
        if split == "train":
            self.annotations = [
                ann for ann in self.annotations if ann.get("split") == "train"
            ]
        elif split == "val":
            self.annotations = [
                ann for ann in self.annotations if ann.get("split") == "val"
            ]
        elif split == "test":
            self.annotations = [
                ann for ann in self.annotations if ann.get("split") == "test"
            ]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx].copy()
        img_path = self.data_dir / "images" / f"{ann['id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        ann["image"] = image

        ann["content"] = ann["visual_sentences"]
        ann["context"] = ann["contextual_sentences"]

        if self.transform:
            ann = self.transform(ann)

        return ann
