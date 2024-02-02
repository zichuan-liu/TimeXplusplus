import torch
import argparse, os
from sklearn import metrics
import numpy as np
from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import *
from txai.vis.vis_saliency import vis_one_saliency
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

is_timex = False


if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv4_consistency import train_mv6_consistency
else:
    from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv6_consistency import train_mv6_consistency

class AttrDict(dict):  
    def __getattr__(self, attr):  
        try:  
            return self[attr]  
        except KeyError:  
            raise AttributeError(f"Attribute '{attr}' not found.")  
      
    def __setattr__(self, attr, value):  
        self[attr] = value 

def naming_convention(args):
    if args.eq_ge:
        name = "bc_eqge_split={}.pt"
    elif args.eq_pret:
        name = "bc_eqpret_split={}.pt"
    elif args.ge_rand_init:
        name = "bc_gerand_split={}.pt"
    elif args.no_ste:
        name = "bc_noste_split={}.pt"
    elif args.simclr:
        name = "bc_simclr_split={}.pt"
    elif args.no_la:
        name = "bc_nola_split={}.pt"
    elif args.no_con:
        name = "bc_nocon_split={}.pt"
    elif args.cnn:
        name = "bc_cnn_split={}.pt"
    elif args.lstm:
        name = "bc_lstm_split={}.pt"
    else:
        name = 'bc_full_split={}.pt'

    if not is_timex:
        name = 'our_'+name
        
    if args.lam != 1.0:
        # Not included in ablation parameters or other, so name it;
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'
    
    return name

class CustomDataset(torch.utils.data.Dataset):  
    def __init__(self, X, times, y):  
        self.X = X
        self.times = times
        self.y = y  
  
    def __len__(self):  
        return len(self.y)  
    
    def __getitem__(self, idx):
        x = self.X[:, idx,:]
        T = self.times[:, idx]
        y = self.y[idx]
        return x, T, y, torch.tensor(idx).long().to(x.device)

def main(args):

    if args.lstm:
        arch = 'lstm'
        tencoder_path = "models/Water_lstm_split={}.pt"
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = "models/Water_cnn_split={}.pt"
    else:
        arch = 'transformer'
        tencoder_path = "models/transformer_split={}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 2,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
    )

    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {'simclr_training':True, 'num_negatives_simclr':64}
    else:
        sim_criterion_cons = EmbedConsistencyLoss(normalize_distance = True)
        sc_expand_args = {'simclr_training':False, 'num_negatives_simclr':64}
    #sim_criterion_cons = EmbedConsistencyLoss(normalize_distance = True)

    if args.no_la:
        sim_criterion = sim_criterion_cons
        selection_criterion = None
    elif args.no_con:
        sim_criterion = sim_criterion_label
        selection_criterion = None
    else: # Regular
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        if args.simclr:
            selection_criterion = simloss_on_val_wboth([cosine_sim_for_simclr, sim_criterion_label], lam = 1.0)
        else:
            selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)

    targs = transformer_default_args
    all_results = {"AUROC": [],
                   "AUPRC": [],
                   "AUP": [],
                   "AUR": [],
                             }
    for i in range(1, 6):
        # if (i == 3):
        #     continue
        D = process_Synth(split_no = i, device = device, base_path = '/TimeX/datasets/water')
        dset = CustomDataset(D['train_loader'].X.to(device) , D['train_loader'].times.to(device), D['train_loader'].y.to(device))
        train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

        val, test = D['val'], D['test']
        # gt_exps = D['gt_exps']

        # Calc statistics for baseline:
        mu = D['train_loader'].X.mean(dim=1)
        std = D['train_loader'].X.std(unbiased = True, dim = 1)

        # Change transformer args:
        targs['trans_dim_feedforward'] = 64
        targs['trans_dropout'] = 0.1
        targs['nlayers'] = 1
        targs['stronger_clf_head'] = False
        targs['norm_embedding'] = True

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
            archtype = arch
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 0.0
        }

        model = TimeXModel(
            d_inp = val[0].shape[-1],
            max_len = val[0].shape[0],
            n_classes = 2,
            n_prototypes = 50,
            gsat_r = 0.5,
            transformer_args = targs,
            ablation_parameters = abl_params,
            loss_weight_dict = loss_weight_dict,
            masktoken_stats = (mu, std),
            tau = 1.0
        )

        model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
        model.to(device)

        if is_timex:
            model.init_prototypes(train = (D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device)))

            if not args.ge_rand_init: # Copies if not running this ablation
                model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

        for param in model.encoder_main.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001)
        
        model_suffix = naming_convention(args)
        spath = os.path.join('models', model_suffix)
        spath = spath.format(i)
        print('saving at', spath)

        # best_model = train_mv6_consistency(
        #     model,
        #     optimizer = optimizer,
        #     train_loader = train_loader,
        #     clf_criterion = clf_criterion,
        #     sim_criterion = sim_criterion,
        #     beta_exp = 1.0,
        #     beta_sim = 1.0,
        #     lam_label = 1.0,
        #     val_tuple = val, 
        #     num_epochs = 200,
        #     save_path = spath,
        #     train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
        #     early_stopping = True,
        #     selection_criterion = selection_criterion,
        #     label_matching = True,
        #     embedding_matching = True,
        #     use_scheduler = False,
        #     bias_weight = 0,
        #     **sc_expand_args
        # )

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)

        X, times, y = test
        index = (y==1)
        X = X[:, index, :]
        times = times[:, index]
        y = y[index]
        test = (X, times, y)

        f1, out = eval_mv4(test, model, masked=True)
        print('Test F1: {:.4f}'.format(f1))

        
        fig, ax = plt.subplots(1, 2, sharex = True, squeeze = False)

        #ax[0,0].set_title('test')
        generated_exps =  out['mask_logits'].transpose(0,1).clone().detach().cpu()
        sampy =  test[2].clone().detach().cpu().numpy()
        pred =  out['pred_mask'].softmax(dim=-1).argmax(dim=-1).clone().detach().cpu().numpy()

        for kk in range(2):
            vis_one_saliency(test[0][:,kk, -1:], generated_exps[:, kk, -1:], ax, fig, col_num = kk)
            print(sampy[kk], pred[kk])
            # ax[0,kk].set_title('y = {:d}, yhat = {:d}', sampy[kk], pred[kk])
        
        #fig.set_size_inches(18.5, 3 * d)
        fig.set_size_inches(18, 5)
        savepdf = "./vis/{}.png".format(i)
        if savepdf is not None:
            plt.savefig(savepdf,dpi=400)
        plt.show()



def print_results(mask_labelss, true_labelss):
    mask_labelss = normalize_exp(mask_labelss)

    if torch.is_tensor(mask_labelss):
        mask_labelss = mask_labelss.cpu().numpy()
    if torch.is_tensor(true_labelss):
        true_labelss = true_labelss.cpu().numpy()

    all_aupoc, all_auprc, all_aup, all_aur = [], [], [], []

    for i in range(mask_labelss.shape[1]):
        mask_label = mask_labelss[:, i, :]
        true_label = true_labelss[:, i, :]

        mask_prec, mask_rec, mask_thres = metrics.precision_recall_curve(
            true_label.flatten().astype(int), mask_label.flatten())
        AUROC = metrics.roc_auc_score(true_label.flatten(), mask_label.flatten())
        all_aupoc.append(AUROC)
        AUPRC = metrics.auc(mask_rec, mask_prec)
        all_auprc.append(AUPRC)
        AUP = metrics.auc(mask_thres, mask_prec[:-1])
        all_aup.append(AUP)
        AUR = metrics.auc(mask_thres, mask_rec[:-1])
        all_aur.append(AUR)
    print('Saliency AUROC: = {:.4f} +- {:.4f}'.format(np.mean(all_aupoc), np.std(all_aupoc) / np.sqrt(len(all_aupoc))))
    print('Saliency AUPRC: = {:.4f} +- {:.4f}'.format(np.mean(all_auprc), np.std(all_auprc) / np.sqrt(len(all_auprc))))
    print('Saliency AUP: = {:.4f} +- {:.4f}'.format(np.mean(all_aup), np.std(all_aup) / np.sqrt(len(all_aup))))
    print('Saliency AUR: = {:.4f} +- {:.4f}'.format(np.mean(all_aur), np.std(all_aur) / np.sqrt(len(all_aur))))

    resutlt_cur = {"AUROC": all_aupoc,
                   "AUPRC": all_auprc,
                   "AUP": all_aup,
                   "AUR": all_aur}
    return resutlt_cur

def normalize_exp(exps):
    norm_exps = torch.empty_like(exps)
    for i in range(exps.shape[1]):
        norm_exps[:,i,:] = (exps[:,i,:] - exps[:,i,:].min()) / (exps[:,i,:].max() - exps[:,i,:].min() + 1e-9)
    return norm_exps

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action = 'store_true', help = 'G = G_E')
    ablations.add_argument('--eq_pret', action = 'store_true', help = 'G_pret = G')
    ablations.add_argument('--ge_rand_init', action = 'store_true', help = "Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action = 'store_true', help = 'Does not use STE')
    ablations.add_argument('--simclr', action = 'store_true', help = 'Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action = 'store_true', help = 'No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action = 'store_true', help = 'No consistency loss - just label')
    # Note if you don't activate any of them, it just trains the normal method
    ablations.add_argument('--lstm', action = 'store_true')
    ablations.add_argument('--cnn', action = 'store_true')
    parser.add_argument('--r', type = float, default = 0.5, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')

    args = parser.parse_args()

    main(args)
