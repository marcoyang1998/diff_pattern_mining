import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class BaseTabularDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path)
        cols = list(self.data.columns)
        assert 'label' == cols[-1], "Col: label is not found in the dataset"
        self.num_feature = len(cols) - 1 # except label col
        self.attribute_cols = cols[:-1]
        self.attribute_data = self.data[self.attribute_cols]
        self.label = self.data['label']

    def __len__(self):
        return len(self.attribute_data)

    def get_model_info(self):
        raise NotImplementedError("Model info method is not implemented!")


class MultiValueTabularDataset(BaseTabularDataset):
    def __init__(self, data_path):
        BaseTabularDataset.__init__(data_path)
        self.feature_value_dict = {} # {feat1: [values], feat2: [values]}
        self.feature_size_dict = {} # {feat1: size, feat2: size}
        self.feature_value2num_dict = {} # {feat1: {val1: 0, val2: 1, ...}, feat2: {val1: 0, ...}}
        for feat in self.attribute_cols:
            value_list = list(set(self.attribute_data[feat]))
            value_list = sorted(value_list)
            self.feature_value_dict[feat] = value_list
            self.feature_size_dict[feat] = len(value_list)
            self.feature_value2num_dict[feat] = {val: i for i,val in enumerate(value_list)}

        self.clarify_data()

    def __getitem__(self, idx):
        transaction = self.attribute_data.iloc[idx]
        data = []
        for i,attr in enumerate(self.attribute_cols):
            data.append(F.one_hot(torch.tensor(transaction[attr]), num_classes=self.feature_size_dict[attr]))
        data = torch.cat(data).float()
        label = torch.from_numpy(np.array(self.label.iloc[idx])).long()
        return data, label

    def get_feature_size_list(self):
        size_list = [self.feature_size_dict[feat] for feat in self.attribute_cols]
        return size_list

    def get_model_info(self):
        return {'feat_size_list': self.get_feature_size_list()}

    def clarify_data(self):
        for feat in self.attribute_data:
            self.attribute_data.replace({feat: self.feature_value2num_dict[feat]}, inplace=True)


class BinaryTabularDataset(BaseTabularDataset):
    def __init__(self, data_path):
        BaseTabularDataset.__init__(self, data_path)
        self.attribute_data = np.array(self.attribute_data)
        self.label = np.array(self.label).reshape(-1,1)

    def get_model_info(self):
        return {'num_feat': self.num_feature}

    def __getitem__(self, idx):
        transaction = torch.from_numpy(self.attribute_data[idx]).float()
        label = torch.from_numpy(self.label[idx]).long()
        return transaction, label


class BinaryTabularTripletDataset(BaseTabularDataset):
    def __init__(self, data_path):
        BaseTabularDataset.__init__(self, data_path)
        
        self.attribute_data = np.array(self.attribute_data)
        self.label = np.array(self.label)

        self.pattern_idx = self.data.index[self.data['label'] == True].tolist()
        self.non_pattern_idx = self.data.index[self.data['label'] == False].tolist()

    def get_model_info(self):
        return {'num_feat': self.num_feature}

    def __getitem__(self, idx):
        anchor = torch.from_numpy(self.attribute_data[idx]).float().view(-1)
        label = self.label[idx]

        if label == 1:
            truthy_idx = np.random.choice(self.pattern_idx, 1)
            while truthy_idx == idx:
                truthy_idx = np.random.choice(self.pattern_idx, 1)
            falsy_idx = np.random.choice(self.non_pattern_idx, 1)
        else:
            truthy_idx = np.random.choice(self.non_pattern_idx, 1)
            while truthy_idx == idx:
                truthy_idx = np.random.choice(self.non_pattern_idx, 1)
            falsy_idx = np.random.choice(self.pattern_idx, 1)

        truthy = torch.from_numpy(self.attribute_data[truthy_idx]).float().view(-1)
        falsy = torch.from_numpy(self.attribute_data[falsy_idx]).float().view(-1)
        #truthy = self.attribute_data[truthy_idx].squeeze()
        #falsy = self.attribute_data[falsy_idx].squeeze()

        return anchor, truthy, falsy


def get_dataloader(args):
    dataset_type = args.dataset_type
    loss_type = args.loss_type

    if dataset_type == 'binary':
        if loss_type == 'classification':
            return BinaryTabularDataset(data_path=args.train_data)
        elif loss_type == 'contrastive':
            return BinaryTabularTripletDataset(data_path=args.train_data)
    elif dataset_type == 'multivalue':
        raise NotImplementedError("Multivalue dataset is not implemented yet!")


if __name__ == '__main__':
    data_path = '/home/v-xiaoyuyang/diff_pattern_mining/data/binary_100000trans_30cols_8pl_0.05noise.csv'
    device = torch.device('cuda:0')
    myDataset = BinaryTabularTripletDataset(data_path=data_path)
    from tqdm import tqdm
    import time
    start = time.time()
    for i in tqdm(range(10000)):
        anchor, truthy, falsy = myDataset[i]
        #anchor.to(device)
        #truthy.to(truthy)
        #falsy.to(falsy)
    end = time.time()
    print(f'Done. Used {end-start} seconds')
