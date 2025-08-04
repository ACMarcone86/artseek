import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from safetensors.torch import load_file


class ArtGraphClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        label_type: str = "id",
        split: str = "train",
        with_image: bool = True,
        transform=None,
    ):
        """Initialize the ArtGraph classification dataset.

        Args:
            data_dir (str | Path): Path to dataset
            label_type (str, optional): Type of label to use ('id', 'name', 'both'). Defaults to "id".
            split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to "train".
            with_image (bool, optional): Whether to include images. Defaults to True.
            transform (_type_, optional): Transform to apply to the dataset. Defaults to None.
        """
        self.data_dir = Path(data_dir)
        self.label_type = label_type
        self.split = split
        self.with_image = with_image
        self.transform = transform

        # Load annotations
        with open(self.data_dir / f"{split}.json", "r") as f:
            self.annotations = json.load(f)

        # Relationships as items
        self.task_class_counts = {}
        self.items = []
        for artwork_id, rels in self.annotations["relationships"].items():
            item = rels.copy()
            for key, value in item.items():
                if key not in self.task_class_counts:
                    self.task_class_counts[key] = {}
                for elem in value:
                    if elem not in self.task_class_counts[key]:
                        self.task_class_counts[key][elem] = 0
                    self.task_class_counts[key][elem] += 1
            item["artwork"] = [artwork_id]
            self.items.append(item)

        # If the split is val, we want to shuffle the items
        if self.split == "val" or self.split == "test":
            rng = random.Random(42)
            rng.shuffle(self.items)

    def id2name(self, node_type, node_id):
        return self.annotations[node_type][node_id]

    def name2id(self, node_type, node_name):
        for node_id, node in self.annotations[node_type].items():
            if node["name"] == node_name:
                return node_id
        return None

    def __len__(self):
        return len(self.annotations["relationships"])

    def __getitem__(self, idx):
        keys = ("artwork", "artist", "genre", "media", "style", "tag")
        item = self.items[idx].copy()

        img_name = self.annotations["artwork"][item["artwork"][0]]
        if self.with_image:
            img_path = self.data_dir / "images" / img_name
            image = Image.open(img_path).convert("RGB")
            item["image"] = image

        for key in keys:
            if not key in item:
                item[key] = []

            new_elems = []
            for elem in item[key]:
                if self.label_type == "name":
                    new_elems.append(self.id2name(key, str(elem)))
                elif self.label_type == "id":
                    new_elems.append(elem)
                elif self.label_type == "both":
                    new_elems.append((elem, self.id2name(key, str(elem))))
            item[key] = new_elems

        if self.transform:
            item = self.transform(item)

        return item
