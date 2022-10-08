from torch.utils.data import Dataset
import pandas as pd
from matplotlib import image

class HackathonDataset(Dataset):
    def __init__(self, type: str == 'train'):
        super(HackathonDataset, self).__init__()

        self.type = type

        data = pd.read_csv('ai_ready/x-ai_data.csv')
        self.data = data[data['split'] == type].T.to_dict()

        for key in self.data:
            file = self.data[key]['filename']
            img = image.imread(f'ai_ready/images/{file}')
            mask = image.imread(f'ai_ready/masks/{file}')

            self.data[key]['img'] = img
            self.data[key]['mask'] = mask

        print('b')

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
