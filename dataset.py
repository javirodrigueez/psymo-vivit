from torch.utils.data import Dataset
import glob, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import VivitImageProcessor
import numpy as np

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Convertir imagen a tensor
])

transform_improved = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])


class GaitDataset(Dataset):
    def __init__(self, data_path, annot_path, trait_name, split='train', transform=None, samples_size=50):
        # Init processor
        self.image_processor = VivitImageProcessor()
        # Load data
        self.videos = []
        vpath = os.path.join(data_path, 'silhouettes/')
        subjects = sorted(glob.glob(f'{vpath}/*'))
        if split == 'train':
            subjects = subjects[:samples_size]
        elif split == 'val':
            idx_test = 251+samples_size
            assert idx_test <= 311
            subjects = subjects[251:idx_test]
        else:
            idx_test = 272+samples_size
            assert idx_test <= 311
            subjects = subjects[251:idx_test]
        for vsubject in subjects:
            videos = sorted(glob.glob(os.path.join(vsubject, '*')))
            videos = [f for f in videos if os.path.isdir(f)]
            self.videos.extend(videos)

        # Load annotations
        self.annotations = []
        df = pd.read_csv(os.path.join(annot_path, 'metadata_labels_v3.csv'))
        trait_name = f'{trait_name}_Label'
        df = df[['ID', trait_name]].sort_values(by='ID').head(samples_size)
        # encode trait_name
        if 'BFI' in trait_name or 'BPAQ' in trait_name:
            mapping = {'Low': 0, 'Normal / Low': 1, 'Normal / High': 2, 'High': 3}
            df[trait_name] = df[trait_name].map(mapping)
        elif 'RSE' in trait_name:
            mapping = {'Low': 0, 'Normal': 1, 'High': 2}
            df[trait_name] = df[trait_name].map(mapping)
        elif 'OFER' in trait_name:
            mapping = {'Low': 0, 'Moderate / Low': 1, 'Moderate / High': 2, 'High': 3}
            df[trait_name] = df[trait_name].map(mapping)
        elif 'DASS' in trait_name:
            mapping = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely Severe': 4}
            df[trait_name] = df[trait_name].map(mapping)
        elif 'GHQ' in trait_name:
            mapping = {'Typical': 0, 'Minor Distress': 1, 'Major Distress': 2}
            df[trait_name] = df[trait_name].map(mapping)

        # expand annotations
        self.annotations = [[i]*48 for i in df[trait_name].tolist()]    # x48 because each annotation is realated with 48 video sequences
        self.annotations = [item for sublist in self.annotations for item in sublist]
        



    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get video frames
        video = self.videos[idx]
        frames = sorted(glob.glob(f'{video}/*.png'))
        video_frames = []
        for frame in frames:
            img = Image.open(frame)
            img_rgb = img.convert('RGB')
            img_trans = transform_improved(img_rgb)
            video_frames.append(img_trans)
        video_fr = torch.stack(video_frames)

        s = video_fr
        max_frames = 16
        if s.size(0) > max_frames:
            # Si la secuencia tiene más de max_frames frames, selecciona max_frames frames de forma uniforme
            indices = torch.linspace(0, s.size(0) - 1, max_frames).long()
            new_sequence = s[indices]
        else:
            # Si la secuencia tiene menos de max_frames frames, agregar padding
            padding = torch.zeros(max_frames - s.size(0), *s.shape[1:])
            new_sequence = torch.cat([s, padding], dim=0).float()
        input_frames = self.image_processor(list(new_sequence), return_tensors="pt", do_rescale=0, offset=0)

        # Get annotations
        label = self.annotations[idx]

        return input_frames, label