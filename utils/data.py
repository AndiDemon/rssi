import pandas as pd
import numpy as np

import torch


from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



class Data(Dataset):
    def __init__(self, data=[]):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)



if __name__=="__main__":
    data = Data()
    print(len(data))



