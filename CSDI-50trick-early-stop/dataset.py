from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
import torch
from torch.utils.data import random_split
import random

import numpy as np
try:
    from sklearn.preprocessing import MinMaxScaler
except:
    print("import wrong")

# 本项目的代码将会固定遮盖住一个window当中的一半内容

class TrainData(Dataset):

    def __init__(self, file_path, test_path, window_length=100):
        self.data = pickle.load(
            open(file_path, "rb")
        )
        length = self.data.shape[0]

        self.test_data = pickle.load(
            open(test_path, "rb")
        )
        self.data = np.concatenate([self.data, self.test_data])
        self.data = torch.Tensor(self.data)
        # 为了避免高斯噪声造成的影响过大，此处将原有的数值全部乘以20
        self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - 100))


    def get_mask(self, observed_mask):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        if random.random() < 0.5:
            # 此时采取策略1，遮盖住0-25,50-75的内容
            mask[length // 4: length // 2, :] = 1
            mask[length - length // 4: , :] = 1
        else:
            # 此时采取策略2，遮盖住25 - 50, 75 - 100的内容
            mask[0 : length // 4, :] = 1
            mask[length // 2: length - length // 4, :] = 1
        return mask


    def __len__(self):
        return len(self.begin_indexes)

    def __getitem__(self, item):
        observed_data = self.data[
            self.begin_indexes[item] :
               self.begin_indexes[item] + self.window_length
        ]
        observed_mask = torch.ones_like(observed_data)
        gt_mask = self.get_mask(observed_mask)
        timepoints = np.arange(self.window_length)
        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints
        }

class TestData(Dataset):

    def __init__(self, file_path,label_path, train_path,window_length=100, get_label=False,window_split=1,strategy = 1):
        self.strategy = strategy
        self.get_label = get_label
        self.data = pickle.load(
            open(file_path, "rb")
        )
        length = self.data.shape[0]
        try:
            self.train_data  = pickle.load(
                open(train_path, "rb")
            )
        except:
            print("train data get wrong !")

        try:
            self.label = pickle.load(
                open(label_path,"rb")
            )
        except:
            print("label get wrong !")
        self.label = torch.LongTensor(self.label)
        self.data = np.concatenate([self.data, self.train_data])
        self.data = torch.Tensor(self.data)
        self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - 100, self.window_length // window_split))

    def __len__(self):
        return len(self.begin_indexes)

    def get_mask(self, observed_mask):
        mask = torch.zeros_like(observed_mask)
        # print("shape of mask is")
        # print(mask.shape)
        # print(mask)
        length = observed_mask.shape[0]
        if self.strategy == 1:
            # 此时采取策略1，遮盖住0-25,50-75的内容
            mask[length // 4: length // 2, :] = 1
            mask[length - length // 4:, :] = 1
        else:
            # 此时采取策略2，遮盖住25 - 50, 75 - 100的内容
            mask[0: length // 4, :] = 1
            mask[length // 2: length - length // 4, :] = 1
        return mask

    def __getitem__(self, item):
        observed_data = self.data[
                        self.begin_indexes[item]:
                        self.begin_indexes[item] + self.window_length
                        ]
        observed_mask = torch.ones_like(observed_data)
        gt_mask = self.get_mask(observed_mask)
        timepoints = np.arange(self.window_length)
        label = self.label[
            self.begin_indexes[item] :
           self.begin_indexes[item] + self.window_length
        ]


        if self.get_label:
            return {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                "label": label
            }
        else:
            return {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
            }

def get_dataloader(train_path, test_path, label_path,batch_size = 32,window_split=1):
    train_data = TrainData(train_path,test_path)
    train_data, valid_data = random_split(
        train_data, [len(train_data) - int(0.1 * len(train_data)) , int(0.1 * len(train_data)) ]
    )

    test_data_strategy_1 = TestData(test_path, label_path, train_path,window_split=window_split,strategy=1)
    test_data_strategy_2 = TestData(test_path, label_path, train_path, window_split=window_split, strategy=2)

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=True)

    test_loader1 = DataLoader(test_data_strategy_1,batch_size=batch_size)
    test_loader2 = DataLoader(test_data_strategy_2,batch_size=batch_size)

    return train_loader, valid_loader, test_loader1, test_loader2


if __name__ == "__main__":
    train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
        "data/Machine/machine-1-1_train.pkl",
        "data/Machine/machine-1-1_test.pkl",
        "data/Machine/machine-1-1_test_label.pkl",
    )
    for batch in test_loader2:
        break
    temp = batch["gt_mask"][23]
    for item in temp:
        print(item)

