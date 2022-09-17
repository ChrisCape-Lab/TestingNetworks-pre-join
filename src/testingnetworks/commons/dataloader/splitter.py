import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.testingnetworks.commons.dataloader.tasker import Tasker


class SimpleDataLoader(Dataset):
    def __init__(self, tasker: Tasker, indexes: list, time_window: int):
        self.tasker = tasker
        self.indexes = indexes
        self.time_window = time_window

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        sample = self.tasker.get_sample(index=idx, time_window=self.time_window)

        return sample


class Splitter:
    def __init__(self, tasker: Tasker, split_proportions: list, data_loading_params: dict, test_tasker: Tasker = None):
        train_proportion = split_proportions[0]
        val_proportion = split_proportions[1]
        assert (train_proportion + val_proportion < 1 and test_tasker is None) or (train_proportion + val_proportion == 1 and test_tasker is not None), \
            'There\'s no space for test samples'
        self.tasker = tasker

        indexes = [i for i in range(0, len(tasker.labels_list))]
        if tasker.data_extractor.is_static and data_loading_params['shuffle']:
            random.shuffle(indexes)

        # Split the data
        if test_tasker is None:
            test_tasker = tasker

            first_split = int(train_proportion * len(indexes))
            second_split = int((train_proportion + val_proportion) * len(indexes))

            train_idx = indexes[:first_split]
            val_idx = indexes[first_split:second_split]
            test_idx = indexes[second_split:]
        else:
            split = int(train_proportion * len(indexes))

            train_idx = indexes[:split]
            val_idx = indexes[split:]

            test_idx = [i for i in range(0, len(test_tasker.labels_list))]

        train = SimpleDataLoader(tasker=tasker, indexes=train_idx, time_window=1)
        self.train = DataLoader(train, shuffle=tasker.data_extractor.is_static,  collate_fn=lambda x: x, num_workers=data_loading_params['num_workers'])

        val = SimpleDataLoader(tasker=tasker, indexes=val_idx, time_window=1)
        self.val = DataLoader(val, shuffle=False,  collate_fn=lambda x: x, num_workers=data_loading_params['num_workers'])

        test = SimpleDataLoader(tasker=test_tasker, indexes=test_idx, time_window=1)
        self.test = DataLoader(test, shuffle=False,  collate_fn=lambda x: x, num_workers=data_loading_params['num_workers'])
