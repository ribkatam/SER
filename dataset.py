
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd



class CustomDataset(Dataset):
    def __init__(self, ann_path, device):  
        self.data =  pd.read_csv(ann_path)
        self.device = device

    def __getitem__(self, idx):
      
        feature_path = self.data.iloc[idx][0]
        label = self.data.iloc[idx][1] 
        feature = torch.load(feature_path, map_location=self.device)
        return feature, torch.tensor(label).long().to(self.device)

    def __len__(self):
        return len(self.data)










