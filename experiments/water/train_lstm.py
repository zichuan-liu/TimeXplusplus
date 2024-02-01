import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.simple import LSTM
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from torch.utils.data import Dataset  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

class AttrDict(dict):  
    def __getattr__(self, attr):  
        try:  
            return self[attr]  
        except KeyError:  
            raise AttributeError(f"Attribute '{attr}' not found.")  
      
    def __setattr__(self, attr, value):  
        self[attr] = value 

class CustomDataset(Dataset):  
    def __init__(self, X, times, y):  
        self.X = X  
        self.times = times  
        self.y = y  
  
    def __len__(self):  
        return len(self.y)  
  
    def __getitem__(self, idx):  
        return self.X[:, idx, :], self.times[:, idx], self.y[idx]  

for i in range(1,6):
    D = process_Synth(split_no = i, device = device, base_path = '/TimeX/datasets/water')
    dataset = CustomDataset(D['train_loader'].X, D['train_loader'].times, D['train_loader'].y) 
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)


    val, test = D['val'], D['test']

    model = LSTM(
        d_inp = val[0].shape[-1],
        n_classes = 2,
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    
    spath = 'models/Water_lstm_split={}.pt'.format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 2,
        num_epochs = 500,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        use_scheduler = False,
        validate_by_step = None,
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, spath)

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))