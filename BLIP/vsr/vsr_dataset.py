
import json
from torch.utils.data import Dataset, DataLoader
import sys
import os
from PIL import Image


class VsrDataset(Dataset):
    def __init__(self, blip_processor):
        super(VsrDataset).__init__()
        self.data = []
        with open('vsr/all_vsr_validated_data.jsonl', "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                self.data.append(j_line)
        self.blip_processor = blip_processor
        self.images_folder = "../../../datasets/vsr/images"
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_number = sample["image"]
        image_path = os.path.join(self.images_folder,image_number)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.blip_processor(image)
        caption = sample["caption"]
        label = sample["label"]
        relation = sample["relation"]
        return image, caption, label, relation


def get_vsr_loader(dataset, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1
    )
    return dataloader


