import os
import json
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):

        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        # =========================
        # LOAD ANNOTATIONS
        # =========================
        with open(self.ann_path, 'r') as f:
            self.ann = json.load(f)

        # current split
        self.examples = self.ann[self.split]

        # =========================
        # BUILD LABEL VOCAB (FIXED)
        # =========================
        self.label2id = self.build_label_vocab(self.ann)

        # =========================
        # TOKENIZE REPORTS
        # =========================
        for i in range(len(self.examples)):
            tokens = tokenizer(self.examples[i]['report'])
            tokens = tokens[:self.max_seq_length]

            self.examples[i]['ids'] = tokens
            self.examples[i]['mask'] = [1] * len(tokens)

    def __len__(self):
        return len(self.examples)

    # =========================
    # BUILD LABEL VOCAB
    # =========================
    def build_label_vocab(self, ann):
        labels = set()

        for split in ann:
            for sample in ann[split]:
                probs = sample.get('Problems', '')

                if probs:
                    items = [p.strip() for p in probs.split(',') if p.strip()]
                    labels.update(items)

        label2id = {label: idx for idx, label in enumerate(sorted(labels))}

        print(f"[INFO] Total unique problem labels: {len(label2id)}")

        return label2id

    # =========================
    # MULTI-HOT ENCODING
    # =========================
    def encode_problems(self, probs):
        vec = np.zeros(len(self.label2id), dtype=np.float32)

        if probs:
            items = [p.strip() for p in probs.split(',') if p.strip()]

            for item in items:
                if item in self.label2id:
                    vec[self.label2id[item]] = 1.0

        # normalize (VERY IMPORTANT)
        if vec.sum() > 0:
            vec = vec / vec.sum()

        return vec

class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]

        image_id = example['id']
        image_path = example['image_path']

        # =========================
        # LOAD IMAGE 1
        # =========================
        try:
            image_1 = Image.open(
                os.path.join(self.image_dir, image_path[0])
            ).convert('RGB')
        except:
            image_1 = Image.new('RGB', (224, 224))

        # =========================
        # LOAD IMAGE 2
        # =========================
        if len(image_path) > 1:
            try:
                image_2 = Image.open(
                    os.path.join(self.image_dir, image_path[1])
                ).convert('RGB')
            except:
                image_2 = image_1
        else:
            image_2 = image_1

        # =========================
        # TRANSFORM
        # =========================
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        image = torch.stack((image_1, image_2), 0)

        # =========================
        # REPORT
        # =========================
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # =========================
        # PROBLEMS (FIXED)
        # =========================
        problem_vec = self.encode_problems(example.get('Problems', ''))
        problem_vec = torch.FloatTensor(problem_vec)

        return image_id, image, problem_vec, report_ids, report_masks, seq_length


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]

        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(
            os.path.join(self.image_dir, image_path[0])
        ).convert('RGB')

        if self.transform:
            image = self.transform(image)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        problem_vec = self.encode_problems(example.get('Problems', ''))
        problem_vec = torch.FloatTensor(problem_vec)

        return image_id, image, problem_vec, report_ids, report_masks, seq_length