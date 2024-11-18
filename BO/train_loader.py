from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, train_data, labels):

        self.train_data = train_data
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        input_features = self.train_data[idx]
        label = self.labels[idx]
        return input_features, label
    
class TestDataset(Dataset):
    def __init__(self, test_data):
        self.test_data = test_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        input_features = self.test_data[idx]
        return input_features