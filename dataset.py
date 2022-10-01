from torch.utils.data import Dataset
import torch


class TSDataset(Dataset):
    def __init__(self, data, labels, is_train=True):
        super(TSDataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        if len(self.data.shape) == 2:
            self.data = self.data.unsqueeze(1)
        self.len, self.channel, self.seq_len = self.data.shape
        self.is_train = is_train
        self.is_train_flag = torch.zeros(self.len, dtype=torch.bool)
        if self.is_train:
            self.is_train_flag = torch.ones(self.len, dtype=torch.bool)
        self.id = torch.arange(0, self.len).long()

    def __getitem__(self, item):
        return self.data[item].type(torch.float32), \
               self.labels[item].type(torch.int64), \
               self.is_train_flag[item], \
               self.id[item]

    def __len__(self):
        return self.len
