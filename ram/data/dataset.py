import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption
import os,glob

import torch
import numpy as np


class finetune_dataset(Dataset):
    def __init__(self, data_id, transform, transform_224, class_num=4585):
        # Load the dataset from Hugging Face
        self.dataset = load_dataset(data_id, split='train')
        
        self.transform = transform
        self.transform_224 = transform_224
        self.class_num = class_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Load the image, caption, and labels from the dataset
        item = self.dataset[index]
        
        # The image is in PIL format directly from the dataset
        image = item['image_path'].convert('RGB')
        image = self.transform(image)

        image_224 = item['image_path'].convert('RGB')
        image_224 = self.transform_224(image_224)

        # Process union_label_id (if available)
        if item.get('union_label_id') is not None:
            num = item['union_label_id']
            image_tag = np.zeros([self.class_num])
            image_tag[num] = 1
            image_tag = torch.tensor(image_tag, dtype=torch.long)
        else:
            image_tag = None

        # Select a random caption (if multiple captions exist)
        caption_index = np.random.randint(0, len(item['caption']))
        caption = pre_caption(item['caption'][caption_index], 30)

        # Process parse_label_id for the caption
        num = item['parse_label_id'][caption_index]
        parse_tag = np.zeros([self.class_num])
        parse_tag[num] = 1
        parse_tag = torch.tensor(parse_tag, dtype=torch.long)

        return image, image_224, caption, image_tag, parse_tag
