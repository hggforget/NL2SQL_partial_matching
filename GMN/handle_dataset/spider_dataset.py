from torch.utils.data import Dataset
import random


class SpiderDataset(Dataset):
    def __init__(self, batch_data_list):
        self.pair_list = batch_data_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        edge_tuple, node_tuple, n_graphs, labels = self.pair_list[idx]

        return edge_tuple, node_tuple, n_graphs, labels

    def shuffle(self):
        random.shuffle(self.pair_list)



