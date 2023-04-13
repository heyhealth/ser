from torch.utils.data import Dataset
import torch


class Dataset_(Dataset):

    def __init__(self, X, y):
        super(Dataset_, self).__init__()
        self.features = X
        self.targets = y

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.targets[index],
                                                                                     dtype=torch.long)

    def __len__(self):
        if len(self.features) == len(self.targets):
            return len(self.targets)
        else:
            raise Exception('features length not equal to targets length!')


