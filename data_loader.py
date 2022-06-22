import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.load_dataset(self.config['dataset'])

    def __getitem__(self, idx):
        return self.rolling_windows[idx, :, :]

    def __len__(self):
        self.rolling_windows.shape[0]
    
    def load_dataset(self, dataset):
        data_dir = "/home/longmeow/Documents/Test_Code/Data_longmeow/{}/".format(self.config["data_dir"])
        self.data = np.load(data_dir + dataset + ".npz")

        if self.mode == 'train':
            data = self.data['train']
        else:
            data = self.data['test']

        # slice training set into rolling windows
        self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                data, self.config["l_win"], axis=0, writeable=True).transpose(0,2,1)


