import torch
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class loader_creation:
    # data_in, k_all, f_all in numpy array
    def __init__(self, data_in, k_all, f_all, length):

        data_in, k_all, f_all = [torch.as_tensor(x, dtype=torch.double).to(device)
                                 if not torch.is_tensor(x) else x
                                 for x in [data_in, k_all, f_all]]

        split_index = max(1, int(0.8*length))

        # splitting to training and testing 
        train_in, k_train, f_train = data_in[0:split_index], k_all[0:split_index], f_all[0:split_index]
        test_in , k_test , f_test  = data_in[split_index:] , k_all[split_index:] , f_all[split_index:]
        
        # passing to dataset to make it iterable
        self.ds_train = customdataset_withKandF(train_in, k_train, f_train)
        self.ds_test  = customdataset_withKandF(test_in, k_test, f_test)
    
    def get_loaders(self, b_size=8, shuffle=True):

        train_loader = DataLoader(self.ds_train, batch_size=b_size, shuffle=shuffle, drop_last=True)
        test_loader  = DataLoader(self.ds_test , batch_size=b_size, shuffle=False  , drop_last=False)
        
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


# #------------------------------------------------------------------------

# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.model_selection import KFold

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Sua classe Dataset permanece essencialmente a mesma
# class CustomDatasetWithKandF(Dataset):
#     def __init__(self, data_in, k, f):
#         # Recomendação: Mantenha os dados na CPU aqui e mova para GPU apenas no loop de treino
#         # para economizar memória VRAM se o dataset for grande.
#         self.data_in = torch.as_tensor(data_in, dtype=torch.float32)
#         self.k = torch.as_tensor(k, dtype=torch.float32)
#         self.f = torch.as_tensor(f, dtype=torch.float32)

#     def __getitem__(self, index):
#         return self.data_in[index].to(device), self.k[index].to(device), self.f[index].to(device)
     
#     def __len__(self):
#         return len(self.data_in)

# # Nova lógica para gerenciar o K-Fold
# def train_kfold(data_in, k_all, f_all, n_splits=10, batch_size=8):
#     dataset = CustomDatasetWithKandF(data_in, k_all, f_all)
    
#     # Inicializa o KFold do scikit-learn
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
#     for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
#         print(f"--- Fold {fold + 1}/{n_splits} ---")
        
#         # Cria Subsets baseados nos índices do K-Fold
#         train_sub = Subset(dataset, train_idx)
#         test_sub  = Subset(dataset, test_idx)
        
#         # Cria os DataLoaders para este fold específico
#         train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
#         test_loader  = DataLoader(test_sub, batch_size=batch_size, shuffle=False)
        
#         # --- Chame sua função de treino aqui ---
#         # model = SuaRedeNeural().to(device)
#         # train_model(model, train_loader, test_loader)
