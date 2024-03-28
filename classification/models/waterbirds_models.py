from typing import Any
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


def resnet50(num_classes=1000):
    model = models.resnet50(pretrained=True)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class WaterbirdsAdversarialLoss:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        self.transform_orign = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def get_loss(self, model, gen_images, batch):
        pred = model(self.transform(gen_images))
        pred = pred * 0.5
        adv_label = 1 - batch['labels']
        # print(F.softmax(pred, dim=1), F.log_softmax(pred, dim=1))
        loss = F.cross_entropy(pred, adv_label)
        return loss

    def get_eval_metrics(self, model, gen_images, batch):
        weights_dtype = gen_images.dtype

        # [-1, 1] -> [0, 1]
        orig_img = batch["pixel_values"] * 0.5 + 0.5
        pred_orig = model(self.transform(orig_img).to(weights_dtype))
        loss_orig = F.cross_entropy(pred_orig, batch['labels'])
        entropy_orig = -(F.softmax(pred_orig, dim=1) * F.log_softmax(pred_orig, dim=1)).sum(dim=1).mean()
        acc_orig = (pred_orig.argmax(dim=1) == batch['labels']).float().mean()

        pred = model(self.transform(gen_images))
        loss = F.cross_entropy(pred, batch['labels'])
        entropy = -(F.softmax(pred, dim=1) * F.log_softmax(pred, dim=1)).sum(dim=1).mean()
        acc = (pred.argmax(dim=1) == batch['labels']).float().mean()

        return {
            "loss_orig": loss_orig.item(),
            "entropy_orig": entropy_orig.item(),
            "acc_orig": acc_orig.item(),
            "loss_adv": loss.item(),
            "entropy_adv": entropy.item(),
            "acc_adv": acc.item(),
        }

