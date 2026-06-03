

from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random


class PromptDataSet(Dataset):
    def __init__(self, train_vi_ir_path_list, val_vi_ir_path_list, phase="train", transform=None):
        self.phase = phase

        if phase == "train":
            self.paths = {
                'vi_ir_A': train_vi_ir_path_list[0],
                'vi_ir_B': train_vi_ir_path_list[1],
            }
            self.paths_gt = {
                'vi_ir_A_gt': train_vi_ir_path_list[2],
                'vi_ir_B_gt': train_vi_ir_path_list[3],
            }
        else:
            self.paths = {
                'vi_ir_A': val_vi_ir_path_list[0],
                'vi_ir_B': val_vi_ir_path_list[1],
            }
            self.paths_gt = {
                'vi_ir_A_gt': val_vi_ir_path_list[0],
                'vi_ir_B_gt': val_vi_ir_path_list[1],
            }

        self.transform = transform

        # Create a list to hold all sample indices grouped by class
        self.class_indices = {}
        for class_key, paths in self.paths.items():
            self.class_indices[class_key] = list(range(len(paths)))
        pass

    def __len__(self):
        if self.phase == "train":
            return 8000
        else:
            return 80

    def __getitem__(self, item):
        if self.phase == "train":
            sample_size = 250

            sample_size = min(
                sample_size,
                len(self.paths['vi_ir_A']),
                len(self.paths['vi_ir_B'])
            )

            vi_index = random.randrange(sample_size)
            ir_index = random.randrange(sample_size)

            image_A_path = self.paths['vi_ir_A'][vi_index]
            image_B_path = self.paths['vi_ir_B'][ir_index]

            image_A_gt_path = self.paths_gt['vi_ir_A_gt'][vi_index]
            image_B_gt_path = self.paths_gt['vi_ir_B_gt'][ir_index]

            class_key = "vi_ir"

        else:
            # ============================================================
            # 验证阶段保持原始配对顺序
            # ============================================================
            image_index = item

            image_A_path = self.paths['vi_ir_A'][image_index]
            image_B_path = self.paths['vi_ir_B'][image_index]

            image_A_gt_path = self.paths_gt['vi_ir_A_gt'][image_index]
            image_B_gt_path = self.paths_gt['vi_ir_B_gt'][image_index]

            class_key = "vi_ir"

        image_A = Image.open(image_A_path).convert(mode='L')
        image_B = Image.open(image_B_path).convert(mode='L')
        image_A_gt = Image.open(image_A_gt_path).convert(mode='L')
        image_B_gt = Image.open(image_B_gt_path).convert(mode='L')

        image_full = image_A

        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B, image_A_gt, image_B_gt, image_full = self.transform(
                image_A,
                image_B,
                image_A_gt,
                image_B_gt,
                image_full
            )

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, image_A_gt, image_B_gt, image_full, class_key, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name = zip(*batch)

        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        images_A_gt = torch.stack(images_A_gt, dim=0)
        images_B_gt = torch.stack(images_B_gt, dim=0)
        images_full = torch.stack(images_full, dim=0)

        return images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name
