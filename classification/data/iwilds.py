import torch
from PIL import Image
import PIL
import wilds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torchvision
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset
from packaging import version
from torchvision import transforms
import random
import json
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

LOCATION_MAP = {1: 0, 78: 1}
LOCATION_MAP_INV = {0: 1, 1: 78}

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ----
prompt_templates_small = ["{}"]


def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)


def read_json(path):
  # with tf.io.gfile.GFile(path) as f:
    return json.load(open(path,'rb'))


def create_detection_map(annotations):
  """Creates a dict mapping IDs to detections."""

  ann_map = {}
  for image in annotations['images']:
    ann_map[image['id']] = image['detections']
  return ann_map



class IWildCamDataset(Dataset):
    def __init__(
        self,
        # cond_data_root,
        tokenizer,
        ## should point to iwildcam training images
        data_root="wilds/iwildcam_v2.0/train",
        ## please find the instance masks as described in iWildCam2021 README https://github.com/visipedia/iwildcam_comp/blob/master/2021/readme.md
        masks_dir = "wilds/iwildcam2021/instance_masks/instance_masks",
        size=512,
        # pred_size=384,
        interpolation="bilinear",
        flip_p=0.5,
        split="train",
        placeholder_token="*",
        center_crop=False,
        negative_prompt="",
        daylight_time=False,
        night_time=False,
        keep_classes=None,
        prompt_templates=prompt_templates_small,
        use_bbox=False,
        keep_location=None
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.split = split
        self.prompt_templates = prompt_templates
        self.use_bbox = use_bbox

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        #self.templates = prompt_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.negative_prompt = negative_prompt


        if use_bbox:
            BOX_ANNOTATION_FILE = "wilds/iwildcam2021/metadata/iwildcam2021_megadetector_results.json"
            # The annotations file contains annotations for all images in train and test
            annotations = read_json(BOX_ANNOTATION_FILE)
            self.detection_map = create_detection_map(annotations)
        

        self.root = data_root
        self.masks_dir = masks_dir
        self.df = pd.read_csv(f'data/iwildcam_v2.0/{split}_subset.csv')

        self.unique_labels = sorted(self.df.y.unique())
        self.label_map = {j:i for i, j in enumerate(self.unique_labels)}
        self.class_names = ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
        self.class_names_to_label = {i:j for i,j in zip(self.class_names, self.unique_labels)}

        if daylight_time or night_time:
            # Extract datetime subcomponents and include in metadata
            self.df['datetime_obj'] = self.df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

            self.df['hour'] = self.df['datetime_obj'].apply(lambda x: int(x.hour))

            daylight_indexes = (self.df["hour"] >= 9) & (self.df["hour"] <= 17)
            night_time_indexes = (self.df["hour"] >= 20) | (self.df["hour"] <= 5)
            if daylight_time and not night_time:
                self.df = self.df.loc[daylight_indexes]

                print(f"KEEPING DAY ONLY")
                print(f"{len(self.df)} samples left")
                #metadata = metadata.loc[daylight_indexes]
            elif not daylight_time and night_time:
                self.df = self.df.loc[night_time_indexes]
                print(f"KEEPING NIGHT ONLY")
                print(f"{len(self.df)} samples left")
                #metadata = metadata.loc[night_time_indexes]

            #self.metadata = metadata


        if keep_classes is not None:
            keep_classes = [self.class_names_to_label[name] for name in keep_classes]
            self.df = self.df.loc[self.df.y.isin(keep_classes)]

        if keep_location is not None:
            print(f"KEEPING LOCATION {keep_location}")
            self.df = self.df.loc[self.df.location_remapped.isin(keep_location)]
            print(f"{len(self.df)} samples left")


        self.labels = [self.label_map[i] for i in self.df.y]
        self.filenames = self.df.filename

        self.locations = self.df.location_remapped.unique()
        self.location_map = {j:i for i, j in enumerate(self.locations)}
        self.location_labels = [self.location_map[i] for i in self.df.location_remapped]
        
        # self.df = pd.read_csv(f'/work/lisabdunlap/DatasetUnderstanding/data/{split}_subset.csv')

        self.samples = [(os.path.join(self.root, i), l) for (i, l) in zip(self.filenames, self.labels)]
        self.image_paths = [os.path.join(self.root, i) for i  in self.filenames]

        self.targets = [l for _, l in self.samples]
        self.classes = list(sorted(self.df.y.unique()))

        self.groups = self.location_labels
        self.class_weights = get_counts(self.labels)
        self.group_names = self.locations




        print(f"Num samples per class {Counter(self.labels)}")


        self.num_images = len(self.image_paths)
        self._length = self.num_images

        #if set == "train":
        #    self._length = self.num_images * repeats



    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        file_path = self.image_paths[i]
        image = Image.open(file_path)
        #width, height = image.size
        #image = image.crop((0, 30, width, height-30))
        image = image.resize((self.size, self.size))

        example["file_path"] = file_path
        example['labels'] = self.labels[i]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.prompt_templates).format(placeholder_string, class_name=self.class_names[self.labels[i]])

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

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        # img_depth = np.array(image_depth).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
            img_depth = img_depth[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        # image = Image.fromarray(img)
        # image = image.resize((self.size, self.size), resample=self.interpolation)
        img = (img / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(img).permute(2, 0, 1)

        file_name = file_path.split('/')[-1][:-4]
        if self.use_bbox:
            detection_annotations = self.detection_map[file_name]
            mask_from_det = np.ones((self.size,self.size))
            image_height, image_width  = mask_from_det.shape[:2]  
            for d in range(len(detection_annotations)):
                xmin, ymin, width, height = detection_annotations[d]['bbox']
                xmin *= image_width
                ymin *= image_height
                width *= image_width
                height *= image_height
                xmin, ymin = int(xmin), int(ymin)
                width, height = int(width), int(height)
                mask_from_det[ymin:ymin+height,xmin:xmin+width] = 0.

            # mask_det = Image.fromarray((mask_from_det[:,:,np.newaxis].repeat(3,2)*255).astype(np.uint8))
            # mask_det_image = mask_det.resize((448, 448))
            example["conditioning"] = torch.from_numpy(mask_from_det)
        else:


            mask_path = os.path.join(self.masks_dir, f'{file_name}.png')
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).resize((self.size,self.size), resample=Image.NEAREST)
                mask = np.array(mask)
                mask[mask > 0] = 1
                mask=torch.from_numpy(mask).float()

                mask= 1-mask

            else:
                mask=torch.ones((self.size,self.size)).float()
                #mask=1-mask

            example["conditioning"] = mask
        


        return example




class Wilds:
    """
    Wrapper for the WILDS dataset.
    """
    def __init__(self, root="wilds", split='train', transform=None):
        self.dataset = wilds.get_dataset(dataset="iwildcam", root_dir=root)#, download=True)
        self.split_dataset = self.dataset.get_subset(split, transform=transform)
        self._metadata_fields = self.split_dataset.metadata_fields
        self.classes = list(range(self.split_dataset.n_classes))
        self.df = pd.DataFrame(self.split_dataset.metadata_array.numpy(), columns=self.split_dataset.metadata_fields)
        self.groups = self.df.location.values
        self.labels = self.targets = self.df.y.values
        self.class_weights = get_counts(self.labels)
        self.samples = [(i, l) for (i, l) in zip(self.df.sequence.values, self.labels)]
        # self.locations = [m[0] for m in self.split_dataset._metadata_array]
        # location = LOCATION_MAP_INV[1] if split == 'test' else LOCATION_MAP_INV[0]
        # self.location_idxs = np.where(np.array(self.locations) == location)[0]
        # self.groups = [location for _ in range(len(self.location_idxs))]

    def __len__(self):
        return len(self.split_dataset)

    def __getitem__(self, idx):
        # map the idx to the location idx (filter out all other locations)
        # idx = self.location_idxs[idx]
        img, label, metadata = self.split_dataset[idx]
        return img, label, self.groups[idx]

class WILDS:
    """
    Specific subset of WILDS containing 6 classes and 2 test locations.
    """
    def __init__(self, root='/wilds/iwildcam_v2.0/train', split='train', transform=None, daylight_time=False, night_time=False, keep_classes=None, keep_location=None):
        self.root = root
        self.df = pd.read_csv(f'data/iwildcam_v2.0/{split}_subset.csv')

        self.transform = transform
        self.unique_labels = sorted(self.df.y.unique())
        self.label_map = {j:i for i, j in enumerate(self.unique_labels)}
        self.class_names = ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
        self.class_names_to_label = {i:j for i,j in zip(self.class_names, self.unique_labels)}

        if daylight_time or night_time:

            self.df['datetime_obj'] = self.df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

            self.df['hour'] = self.df['datetime_obj'].apply(lambda x: int(x.hour))


            daylight_indexes = (self.df["hour"] >= 9) & (self.df["hour"] <= 17)
            night_time_indexes = (self.df["hour"] >= 20) | (self.df["hour"] <= 5)
            if daylight_time and not night_time:
                self.df = self.df.loc[daylight_indexes]
                #metadata = metadata.loc[daylight_indexes]
            elif not daylight_time and night_time:
                self.df = self.df.loc[night_time_indexes]
                #metadata = metadata.loc[night_time_indexes]

            #self.metadata = metadata


        if keep_classes is not None:
            keep_classes = [self.class_names_to_label[name] for name in keep_classes]
            self.df = self.df.loc[self.df.y.isin(keep_classes)]

        if keep_location is not None:
            self.df = self.df.loc[self.df.location_remapped.isin(keep_location)]

        self.labels = [self.label_map[i] for i in self.df.y]
        self.filenames = self.df.filename

        self.locations = self.df.location_remapped.unique()
        self.location_map = {j:i for i, j in enumerate(self.locations)}
        self.location_labels = [self.location_map[i] for i in self.df.location_remapped]
        
        # self.df = pd.read_csv(f'/work/lisabdunlap/DatasetUnderstanding/data/{split}_subset.csv')

        self.samples = [(os.path.join(root, i), l) for (i, l) in zip(self.filenames, self.labels)]
        self.targets = [l for _, l in self.samples]
        self.classes = list(sorted(self.df.y.unique()))

        self.groups = self.location_labels
        self.class_weights = get_counts(self.labels)
        self.group_names = self.locations

        print(f"Num samples per class {Counter(self.labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        location = self.location_labels[idx]
        ## TODO REMOVE img_path
        return img, label, location

    def inspect_location(self, location):
        assert location in self.locations
        location_df = self.df[self.df.location_remapped == location]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        idx = np.random.choice(list(range(len(location_df))))
        location_df['y'].value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title(f'Location {location} (n={len(location_df)}) class counts')
        axs[1].imshow(Image.open(os.path.join(self.root, location_df.iloc[idx].filename)))
        axs[1].set_title(f'Location {location} (n={len(location_df)}) class {location_df.iloc[idx].y} (idx={idx})')
        axs[1].axis('off')
        plt.show()

    def inspect_class(self, class_idx):
        assert class_idx in self.classes
        class_df = self.df[self.df.y == class_idx]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        idx = np.random.choice(list(range(len(class_df))))
        class_df['location_remapped'].value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title(f'Class {class_idx} (n={len(class_df)}) location counts')
        axs[1].imshow(Image.open(os.path.join(self.root, class_df.iloc[idx].filename)))
        axs[1].set_title(f'Class {class_idx} (n={len(class_df)}) location {class_df.iloc[idx].location_remapped} (idx={idx}) ({class_df.iloc[idx].filename})')
        axs[1].axis('off')
        plt.show()

    @staticmethod
    def eval(y_pred, y_true, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )

        return results, results_str

class WILDSDiffusion(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.df = pd.DataFrame({'img_filename': self.samples, 'y': self.targets})
        self.class_names = ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
        # self.class_names = ['background', 'cattle', 'elephant', 'imapala', 'zebra', 'giraffe', 'dik-dik']
        # self.classes = [int(c) for c in self.classes] 
        self.labels = self.targets
        self.classes = [0, 24, 32, 36, 48, 49, 52]
        # self.class_map = {j:i for i, j in enumerate(self.classes)}
        self.groups = [1000] * len(self.samples)
        self.class_weights = torch.tensor(get_counts(self.labels))
        self.group_names = [1000]
        self.targets = [l for _, l in self.samples]
    

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.groups[idx]

    def inspect_class(self, class_idx):
        class_idxs = np.where(np.array(self.labels) == self.class_map[class_idx])[0]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        idx = np.random.choice(class_idxs)
        filename, label = self.samples[idx]
        ax.imshow(Image.open(filename))
        ax.set_title(f'Class {class_idx} (n={len(class_idxs)}) (idx={idx})')
        ax.axis('off')
        plt.show()

def wilds_eval(y_pred, y_true, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )

        return results, results_str