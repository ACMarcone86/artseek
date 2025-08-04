from pathlib import Path
import json
from torch.utils.data import Dataset
from PIL import Image


class PaintingFormDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", transform=None):
        """Initialize the PaintingForm dataset.

        Args:
            data_dir (str | Path): Path to dataset directory
            split (str): Dataset split ('train', 'val', 'test'). Defaults to "train".
            transform: Transform to apply to the dataset. Defaults to None.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load annotations from JSON file
        if split == "train":
            with open(self.data_dir / f"train_samples_tuning.json", "r") as f:
                self.annotations = json.load(f)
        elif split == "test":
            with open(self.data_dir / f"formalanalysis_test_gemini.json", "r") as f:
                annotations_gemini = json.load(f)
            with open(self.data_dir / f"formalanalysis_test_gpt.json", "r") as f:
                annotations_gpt = json.load(f)
            for gemini_sample, gpt_sample in zip(annotations_gemini, annotations_gpt):
                assert gemini_sample["image"] == gpt_sample["image"]
                gemini_sample["formal_analysis_gemini"] = gemini_sample[
                    "formal_analysis"
                ]
                gemini_sample["formal_analysis_gpt"] = gpt_sample["formal_analysis"]
                del gemini_sample["formal_analysis"]
                gemini_sample["image"] = "images/" + gemini_sample["image"]
            self.annotations = annotations_gemini

    def __len__(self):
        """Return the total number of samples"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): Index of the sample

        Returns:
            dict: Sample containing image and annotations
        """
        # Get annotation for index
        ann = self.annotations[idx].copy()

        # Load image from path
        img_path = self.data_dir / "art_images_data" / ann["image"]
        image = Image.open(img_path).convert("RGB")
        ann["image"] = image

        # Apply transforms if specified
        if self.transform:
            ann = self.transform(ann)

        return ann
