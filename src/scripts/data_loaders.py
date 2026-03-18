import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from pathlib import Path

class AirfoilDataset(Dataset):

    """
    Defining this class so that we can keep track of airfoil names, geometry describers, conditions, and performance coeffs.

    Args:
        geometry (torch.Tensor): Tensor of all the geometry of airfoils, as a tensor
        cond (torch.Tensor): Conditions of all the sims, as a tensor
        perf_coeffs (torch.Tensor): Performance coeffs of all sims, as a tensor
        names (list): Names of the airfoils, as a list
    """
    def __init__(self, geometry:(torch.Tensor), cond:(torch.Tensor), perf_coeffs:(torch.Tensor), names:(list[str])):

        self.geometry = geometry
        self.cond = cond
        self.perf_coeffs = perf_coeffs
        self.names = names

    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, idx):

        return {
            'geometry': self.geometry[idx],
            'cond': self.cond[idx],
            'perf_coeffs': self.perf_coeffs[idx],
            'name': self.names[idx]
        }
    
def get_dataset(df:(pd.DataFrame), loc_geometry:(list[int]), loc_cond:(list[int]), loc_perf_coeffs:(list[int]), loc_names:(int)):

    """
    Given the specific columns of the dataframe, get the dataset.

    Args:
        loc_geometry (list,int): Integer list of the columns for geometry, must be inclusive
        loc_cond (list,int): Integer list of the columns for conditions, must be inclusive
        loc_perf_coeffs (list,int): Integer list of the columns for performance coefficients, must be inclusive
        loc_names (int): Integer of the column of names
    
    Returns:
        tensor_geometry (torch.tensor): A tensor of the geometries
        tensor_cond (torch.tensor): A tensor of the conditions
        tensor_perf_coeffs (torch.tensor): A tensor of the performance coefficients
        list_names (list[str]): A list of names as strings
    """

    tensor_geometry = torch.tensor((df.iloc[:, loc_geometry[0]:loc_geometry[1]+1]).to_numpy(), dtype=torch.float32)
    tensor_cond = torch.tensor((df.iloc[:, loc_cond[0]:loc_cond[1]+1]).to_numpy(), dtype=torch.float32)
    tensor_perf_coeffs = torch.tensor((df.iloc[:, loc_perf_coeffs[0]:loc_perf_coeffs[1]+1]).to_numpy(), dtype=torch.float32)
    list_names = (df.iloc[:,loc_names]).to_list()
    
    return tensor_geometry, tensor_cond, tensor_perf_coeffs, list_names
    
def get_dataloaders(ds:(AirfoilDataset), cfg_loader:(dict), seed:(int)= 31):

    """
    Gets the dataloaders for train, test, validate datasets.

    Args:
        ds (AirfoilDataset): The entire dataset as a AirfoilDataset class.
        cfg_loader (dict):
            - 'n_epoch' (int): How many epochs
            - 'n_train' (int): How many samples to train with, each epoch (n_val will be inferred from this)
            - 'n_test' (int): How many samples to test on, after all the epochs are done
            - 'train_batch' (int): The batch size of each training run, must divide n_train
        seed (int): Seed to separate the datasets, def = 31

    Returns:
        dl_train (DataLoader): Dataloader of the training dataset, with batch size = train_batch
        dl_val (DataLoader): Dataloader of the validation dataset, with batch size = number of validation samples / epoch
        dl_test (DataLoader): Dataloader of the test dataset, with the batch size = n_test
    """

    # Take in the values from the dict
    n_epoch = cfg_loader.get('n_epoch')
    n_train = cfg_loader.get('n_train')
    n_test = cfg_loader.get('n_test')
    train_batch = cfg_loader.get('train_batch')

    # Infer the amount of n_val, ensure completeness
    n_val = int(int(len(ds) - n_epoch*n_train - n_test)/n_epoch)
    assert n_val % n_epoch == 0, f"Something is wrong with the sample allocation, n_val (inferred as: int(len(ds) - n_epoch*n_train - n_test)): {n_val} is not divisible by n_epoch: {n_epoch}"

    # Define the generator for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Separate the dataset into test and temporary
    ds_test, ds_temp = random_split(ds, [n_test, n_epoch*(n_train+n_val)], generator=generator) 

    # Separate the temp dataset into train and validation
    ds_train, ds_val = random_split(ds_temp, [n_epoch*n_train, n_epoch*n_val], generator=generator)

    # Assert the batch size of each training run fully divides total training runs each epoch
    assert n_train % train_batch == 0, f"The total training sample amount ({n_train}) at each epoch must be divisible by the batch size ({train_batch})."

    """
    Return the dataloaders such that
    - Training dataloader has the correct batch size
    - Validation dataloader only has samples the equal number of epochs
    - Test dataloader is a giant single batch
    """
    dl_train = DataLoader(ds_train, batch_size=train_batch, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=int((len(ds_val)/n_epoch)), shuffle= True)
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=True) 

    return dl_train, dl_val, dl_test

def create_dataloaders(cfg_data, cfg_loader):

    data_path = Path(cfg_data['data_path']).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data at: {data_path}")

    df = pd.read_csv(data_path)

    df = df.drop(['N1', 'N2'], axis=1) # Remove N1 and N2 since all the airfoils are subsonic

    df['Reynolds'] = df['Reynolds'] / cfg_data.get('reynolds_norm_factor', 1000000) # Normalize the Reynolds feature, too dominant

    geom, cond, perf, names= get_dataset(df, 
                                        loc_geometry= cfg_data['loc_geometry'], 
                                        loc_cond= cfg_data['loc_cond'], 
                                        loc_perf_coeffs= cfg_data['loc_perf_coeffs'], 
                                        loc_names= cfg_data['loc_names']) # Get the necessary stuff for the dataset

    ds = AirfoilDataset(geom, cond, perf, names)

    del df, geom, cond, perf, names # Delete these to preserve memory

    dl_train, dl_val, dl_test = get_dataloaders(ds=ds, cfg_loader=cfg_loader)

    dataloaders = [dl_train, dl_val, dl_test]; del dl_train, dl_val, dl_test

    return dataloaders

