import torch
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class loader_creation():
    # data_in, k_all, f_all in numpy array
    def __init__(self, data_in, k_all, f_all, length):

        data_in, k_all, f_all = [torch.as_tensor(x, dtype=torch.double).to(device)
                                 if not torch.is_tensor(x) else x
                                 for x in [data_in, k_all, f_all]]

        split_index = max(1, int(0.8*length))

        # splitting to training and testing 
        train_in = data_in[0:split_index]
        test_in  = data_in[split_index:]
        k_train  = k_all[0:split_index]
        f_train  = f_all[0:split_index]
        k_test   = k_all[split_index:]
        f_test   = f_all[split_index:]
        
        # passing to dataset to make it iterable
        self.ds_train = customdataset_withKandF(train_in, k_train, f_train)
        self.ds_test  = customdataset_withKandF(test_in, k_test, f_test)
    
    def get_loaders(self, b_size=8, shuffle=True):

        train_loader = DataLoader(self.ds_train, batch_size=b_size, shuffle=shuffle, drop_last=True)
        test_loader  = DataLoader(self.ds_test , batch_size=b_size, shuffle=False, drop_last=False)
        
        return train_loader, test_loader


class customdataset_withKandF(Dataset):
    # A class to load a custom dataset with k and f.
    def __init__(self, data_in, k, f):
        self.data_in = data_in
        self.k       = k
        self.f       = f 

    def __getitem__(self, index):
        return self.data_in[index], self.k[index], self.f[index]
     
    def __len__(self):
        # Returns size of the whole training dataset.
        return len(self.data_in)
