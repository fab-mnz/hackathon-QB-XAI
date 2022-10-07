from torch.utils.data import Dataset


class HackathonDataset(Dataset):
    def __init__(self):
        super(HackathonDataset, self).__init__()

        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
