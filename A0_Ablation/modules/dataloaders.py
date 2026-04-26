import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_name = args.dataset_name

        # ===== Transforms =====
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            ])

        # ===== Dataset selection =====
        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(
                args, tokenizer, split, transform=self.transform
            )
        else:
            self.dataset = MimiccxrSingleImageDataset(
                args, tokenizer, split, transform=self.transform
            )

        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def collate_fn(self, batch):
        """
        Expected dataset output:
        (image_id, image, problem_vec, report_ids, report_mask, seq_len)
        """

        images_id, images, problem_vec, reports_ids, reports_masks, seq_lengths = zip(*batch)

        # ===== Stack tensors =====
        images = torch.stack(images, 0)
        problem_vec = torch.stack(problem_vec, 0)

        batch_size = len(reports_ids)
        max_len = max(seq_lengths)

        pad_idx = self.tokenizer.pad_idx

        # ===== Initialize padded tensors =====
        targets = torch.full(
            (batch_size, max_len),
            fill_value=pad_idx,
            dtype=torch.long
        )

        target_masks = torch.zeros(
            (batch_size, max_len),
            dtype=torch.float
        )

        # ===== Fill tensors =====
        for i in range(batch_size):
            seq_len = len(reports_ids[i])

            targets[i, :seq_len] = torch.tensor(reports_ids[i], dtype=torch.long)
            target_masks[i, :seq_len] = torch.tensor(reports_masks[i], dtype=torch.float)

        return (
            images_id,
            images,
            problem_vec,
            targets,
            target_masks,
            seq_lengths
        )