import numpy as np
import torch
import pandas as pd
import os 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from .augmentations import RotateColsRandomly, GaussianNoise
import torchvision

class __FlightsDataset(Dataset):

    def __init__(self, root):
        """
        DESC
        ---
        Initializes a Dataset class for the maneuver-id data. Reads the data
        from .tsv files into numpy tensors
        ---
        INPUTS
        ---
        root (str): folder path that includes the two subdirectories called
        'good' and 'bad'. These directories should include only .tsv files
        ---
        RETURN
        ---
         __FlightsDataset instance
        """
        # declaring paths
        self.folder_names = ['good', 'bad']

        # declare arrays to append the data to
        self.X, self.Y = [], []

        # iterate through the two folders
        for folder_name in self.folder_names:
            # iterate through the .tsv files in the two folders
            for file_name in os.listdir(os.path.join(root, folder_name)):

                # join path and read the tsv
                path = os.path.join(root, folder_name, file_name)
                X = pd.read_csv(path, sep = '\t', index_col=0)

                # drop two unecessary columns
                X = X.drop(columns=['time (sec)'])
                X = X.to_numpy(dtype='float32')

                # check for nan
                if np.isnan(X).any():
                  continue

                # append X and Y to arrays
                self.X.append(X)
                self.Y.append(1 if folder_name == 'good' else 0)

        # to numpy
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        # rotate these columns
        X_POS_COL, Y_POS_COL, X_VEL_COL, Y_VEL_COL = 0, 1, 3, 4
        # add gaussian noice with each columns stdev
        NOICE_FACTOR    = 0.05

        # compose transformations
        self.transforms = torchvision.transforms.Compose([
            RotateColsRandomly(X_POS_COL, Y_POS_COL),
            RotateColsRandomly(X_VEL_COL, Y_VEL_COL),
            RotateColsRandomly(factor = NOICE_FACTOR),
        ])
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # read X and Y
        X = self.X[idx]
        Y = self.Y[idx]

        X = torch.tensor(X)
        Y = torch.tensor(Y).long()

        X = self.transforms(X)

        # cast to tensor and return
        return X, torch.tensor(Y).long()
    
    def collate_fn(batch):
        
        # divide to x and y batches
        batch_x = [x for x, _ in batch]
        batch_y = [y for _, y in batch]

        # pad the x's
        batch_x_pad = pad_sequence(batch_x, padding_value = 0, batch_first = True)

        # calculate lengths of x's because of padding
        # available for use with lstm models
        lengths_x   = [len(_) for _ in batch_x]

        # collate the data
        return batch_x_pad, torch.tensor(batch_y), torch.tensor(lengths_x)
        
def DataLoaders(root, batch_size = 8):

    """
    DESC
    ---
    A function that creates three dataloaders: train, val, and test.

    This function uses a seed to standardize the test split to ensure
    all models trained are evaluated on the same, unseen held out test data.

    IMPORTANT !!!!

    do not change this code and do not initialize any data loaders
    in any other way.

    This data loader ensures that every single team member has the
    same held out test dataset.

    ---
    INPUTS
    ---
    root (str): folder path that includes the two subdirectories called
    'good' and 'bad'. These directories should include only .tsv files

    batch_size (int, optional): the batch size for the three dataloaders

    ---
    RETURN
    ---
    train_loader, val_loader, test_loader

    """
    dataset = __FlightsDataset(root)

    train_percentage = 0.50
    val_percentage   = 0.25
    # test percentage is inferred

    N = len(dataset) 
    train_size = int(N * train_percentage)
    val_size = int(N * val_percentage)
    test_size = N - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, collate_fn=__FlightsDataset.collate_fn)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True, collate_fn=__FlightsDataset.collate_fn)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=True, collate_fn=__FlightsDataset.collate_fn,)
    
    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    return train_loader, val_loader, test_loader
