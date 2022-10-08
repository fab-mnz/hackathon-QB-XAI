from torch.utils.data import Dataset
import pandas as pd

class HackathonDataset(Dataset):
    def __init__(self, type: str == 'train'):
        super(HackathonDataset, self).__init__()

        self.type = type

        data = pd.read_csv('ai_ready/x-ai_data.csv')
        self.data = data[data['split'] == type]

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.data)
