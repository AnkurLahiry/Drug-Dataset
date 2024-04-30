from torch.utils.data import Dataset, DataLoader
import torch

class RatingDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.column_names = ["text" ,"label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.texts[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text_tensor, label_tensor
        


class RatingSetfitDaataset(Dataset):
    def __init__(self, data):
        # Add word_count during initialization
        self.data = [{"text": text, "label": label, "word_count": len(text.split())} for text, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
