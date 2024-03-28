import os
import logging

import numpy as np
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from wilds import get_dataset



PROMPT_TEMPLATES = [
    "{}",
]

DATA_ROOT = '/scratch/atanov/data/'


class WaterbirdsDataset(Dataset):
    def __init__(
            self,
            data_root,
            size=512,
            tokenizer=None,
            placeholder_token="*",
            negative_prompt="",
            prompt_templates=PROMPT_TEMPLATES,
            classes=[0, 1],
            num_samples=None,
            split="train",
            places=[0, 1]
        ) -> None:

        self.data_root = data_root
        self.waterbirds = get_dataset(dataset="waterbirds", download=True, root_dir=data_root)
        self.size = size
        self.tokenizer = tokenizer
        self.prompt_templates = prompt_templates
        self.placeholder_token = placeholder_token
        self.negative_prompt = negative_prompt
        self._classes = classes
        # from ALIA
        self._class_names = ["Landbird", "Waterbird"]

        self.transform_mask = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

        # choose only the classes we want
        mask = np.isin(self.waterbirds._y_array.numpy(), classes)
        # choose only the places we want
        mask = np.logical_and(mask, np.isin(self.waterbirds._metadata_array[:, 0].numpy(), places))

        # choose the split
        if split == "train":
            mask = np.logical_and(mask, self.waterbirds._split_array == 0)
        elif split == "val":
            mask = np.logical_and(mask, self.waterbirds._split_array == 1)
        elif split == "test":
            mask = np.logical_and(mask, self.waterbirds._split_array == 2)
        else:
            raise ValueError(f"Unknown split {split}")

        self._valid_indices = np.where(mask)[0]

        if num_samples is not None:
            indices = np.random.permutation(len(self._valid_indices))
            self._valid_indices = self._valid_indices[indices[:num_samples]]

        logging.info(f"Found {len(self._valid_indices)} images for classes {classes} out of {len(self.waterbirds)} total images")

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        example = {}

        idx = self._valid_indices[idx]
        image, label, _ = self.waterbirds[idx]

        assert label in self._classes

        image = image.resize((self.size, self.size), resample=PIL.Image.LANCZOS)
        image = np.array(image.convert("RGB"))
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        example['pixel_values'] = image
        example['labels'] = label

        # HACK: this is a hack to get the mask path, ideally override the getitem method of waterbirds
        img_path = self.waterbirds._input_array[idx]
        mask_path = img_path.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.waterbirds.data_dir, mask_path)
        mask = PIL.Image.open(mask_path)
        example['conditioning'] = 1 - self.transform_mask(mask)

        text = np.random.choice(self.prompt_templates).format(self.placeholder_token, class_name=self._class_names[label])

        if self.tokenizer is None:
            return example

        example["prompt"] = text

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["neg_input_ids"] = self.tokenizer(
            self.negative_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example



