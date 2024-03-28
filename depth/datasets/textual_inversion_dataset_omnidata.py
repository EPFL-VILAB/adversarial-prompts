from torchvision import transforms
from PIL import Image
import PIL
import numpy as np
import torch
from packaging import version
import torchvision.transforms.functional as tvf
import torchvision.transforms as tf

from torch.utils.data import Dataset, Subset, DataLoader


indoor_templates_small2 = [
    "{}, photorealistic, photo, 3D, highly detailed",
    "{}, bright, photorealistic, photo, 3D, highly detailed",
    "{}, dark, photorealistic, photo, 3D, highly detailed",
]


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

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        size=512,
        pred_size=384,
        repeats=100,
        interpolation="bilinear",
        flip_p=0.5,
        center_crop=False,
        negative_prompt="",
        num_new_tokens=4,
        data_paths_file=None,
        add_default_prompt=False,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.pred_size = pred_size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.num_new_tokens = num_new_tokens
        self.data_paths_file = data_paths_file
        self.add_default_prompt = add_default_prompt


        assert len(self.data_paths_file) > 1
        with open(self.data_paths_file[0], 'r') as f:
            data = f.readlines()
        train_image_paths = [x.strip().replace('depth_zbuffer', 'rgb') for x in data]

        with open(self.data_paths_file[1], 'r') as f:
            data = f.readlines()
        val_image_paths = [x.strip().replace('depth_zbuffer', 'rgb') for x in data]

        self.image_paths = train_image_paths + val_image_paths

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.train_indices = list(range(0, len(train_image_paths)))
        self.val_indices = list(range(len(train_image_paths), len(self.image_paths)))

        self._length = self.num_images * repeats
        self.train_indices = self.train_indices * repeats
        self.val_indices = self.val_indices * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = indoor_templates_small2
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.negative_prompt = negative_prompt

        self.placeholder_token_seq = [f'<placeholder_token_{i}>' for i in range(self.num_new_tokens)]
        

    def __len__(self):
        return self._length

    def get_train_dataloader(self, batch_size, shuffle, num_workers):
        train_dataset = Subset(self, self.train_indices)
        return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def get_validation_dataloader(self, batch_size, shuffle, num_workers):
        val_dataset = Subset(self, self.val_indices)
        return DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    def __getitem__(self, i):
        example = {}

        image = Image.open(self.image_paths[i])
        image_depth_file = self.image_paths[i].replace('rgb','depth_zbuffer')
        mask_file = self.image_paths[i].replace('rgb','mask_valid')
        image_depth = Image.open(image_depth_file)
        image_mask = Image.open(mask_file)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        text = ','.join(self.placeholder_token_seq)
        if self.add_default_prompt:
            text += ',photo,highly detailed,photorealistic'

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

        img = np.array(image).astype(np.uint8)

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

        image = Image.fromarray(img)
        image = image.resize((self.pred_size, self.pred_size), resample=self.interpolation)

        flip_img = torch.rand(1) < self.flip_p
        if flip_img: image = tvf.hflip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["rgb"] = torch.from_numpy(image).permute(2, 0, 1)  # rgbs are not used, only need labels

        image_depth = image_depth.resize((self.size, self.size), resample=self.interpolation)
        if flip_img: image_depth = tvf.hflip(image_depth)
        image_depth = np.array(image_depth).astype(np.float32)
        image_depth = image_depth[np.newaxis,:,:]
        image_depth = np.concatenate([image_depth, image_depth, image_depth], axis=0)
        example["gt"] = torch.from_numpy(np.clip(image_depth/(2**16-1),0.,1.))

        inv_gt_norm = (1-example["gt"]).float().numpy()
        image_mask = image_mask.resize((self.size, self.size), resample=self.interpolation)
        mask = np.array(image_mask).astype(bool)
        mask = np.repeat(mask[None,...],3,0)

        inv_gt_norm[mask] = (inv_gt_norm[mask] - inv_gt_norm[mask].min()) / (inv_gt_norm[mask].max() - inv_gt_norm[mask].min())
        inv_gt_norm[~mask] = 0.
        depth_image_inv = torch.from_numpy(inv_gt_norm)
        example["conditioning"] = depth_image_inv

        image_mask_resized = image_mask.resize((self.pred_size, self.pred_size), resample=self.interpolation)
        if flip_img: image_mask_resized = tvf.hflip(image_mask_resized)
        example["mask"] = tf.ToTensor()(image_mask_resized)

        return example
