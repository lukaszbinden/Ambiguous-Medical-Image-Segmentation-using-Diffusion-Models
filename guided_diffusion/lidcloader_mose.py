import numpy as np
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import os
from pathlib import Path
import random
import torch


class lidc_Dataloader():
    def __init__(self, data_folder, transform_train, transform_test, label_range='all'):
        self.train_ds = lidc_Dataset(os.path.join(data_folder, 'train')
                                     , transform_train, label_range, test_flag=False)
        self.val_ds = lidc_Dataset(os.path.join(data_folder, 'val'),
                                   transform=None, test_flag=True)
        self.test_ds = lidc_Dataset(os.path.join(data_folder, 'test'),
                                    transform_test, test_flag=True)

        # self.train = DataLoader(self.train_ds, shuffle=True, batch_size=exp_config.train_batch_size,
        #                         drop_last=True, pin_memory=True, num_workers=exp_config.num_w)
        #
        # self.validation = DataLoader(self.val_ds, shuffle=False, batch_size=exp_config.val_batch_size,
        #                         drop_last=True, pin_memory=True, num_workers=exp_config.num_w)
        #
        # self.test = DataLoader(self.test_ds, shuffle=False, batch_size=exp_config.test_batch_size,
        #                         drop_last=False, pin_memory=True, num_workers=exp_config.num_w)


class lidc_Dataset(Dataset):
    def __init__(self, data_file, transform=None, label_range='all', test_flag=False):

        self.img_file = sorted((Path(data_file) / 'images').iterdir())
        self.label_file = sorted((Path(data_file) / 'labels').iterdir())
        self.transform = transform
        self.label_range = label_range
        self.test_flag = test_flag
        if self.label_range != 'all':
            print(label_range)

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, index: int):

        x = np.load(self.img_file[index])
        y = np.load(self.label_file[index])
        if self.label_range != 'all':
            y = y[:, :, self.label_range]
        # change y from H,W,N to N,H,W
        # y = y.transpose(2,0,1)
        # Do normalization and add first channel to x
        # x = np.expand_dims((x - x.mean()) / x.std(), 0).astype(np.float32)
        x = x + 0.5

        image = torch.tensor(x)
        image = torch.unsqueeze(image, 0)
        image = torch.cat((image, image, image, image), 0)
        x = image

        if self.test_flag:
            # use full y masks for testing
            y = torch.tensor(y)
        else:
            y = y[random.randint(0, 3)]
            y = torch.unsqueeze(torch.tensor(y), 0)

        sample = {
            'image': x,
            'label': y,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # y_prob = np.ones(y.shape[0], dtype=np.float32) / y.shape[0]
        # return sample['image'], sample['label'], y_prob,
        return sample['image'], sample['label']
