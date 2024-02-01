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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

is_timex = False

if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv4_consistency import train_mv6_consistency
else:
    from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv6_consistency import train_mv6_consistency


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
    else:
        name = 'bc_full_split={}.pt'

    if not is_timex:
        name = 'our_'+name
        
    if args.lam != 1.0:
        # Not included in ablation parameters or other, so name it;
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'
    
    return name

def main(args):

    tencoder_path = "./models/transformer_new2_split={}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 4,
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
        if (i == 4):
            continue
        D = process_Synth(split_no = i, device = device, base_path = '/TimeX/datasets/LowVarDetect')
        dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))

        train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

        val, test = D['val'], D['test']
        gt_exps = D['gt_exps']

        # Calc statistics for baseline:
        mu = D['train_loader'].X.mean(dim=1)
        std = D['train_loader'].X.std(unbiased = True, dim = 1)

        # Change transformer args:
        targs['trans_dim_feedforward'] = 32
        targs['trans_dropout'] = 0.1
        targs['nlayers'] = 1
        targs['norm_embedding'] = True

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        model = TimeXModel(
            d_inp = 2,
            max_len = 200,
            n_classes = 4,
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

        optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-3, weight_decay = 0.0001)
        
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
        #     beta_exp = 2.0,
        #     beta_sim = 1.0,
        #     lam_label = 1.0,
        #     val_tuple = val, 
        #     num_epochs = 100,
        #     save_path = spath,
        #     train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
        #     early_stopping = True,
        #     selection_criterion = selection_criterion,
        #     label_matching = True,
        #     embedding_matching = True,
        #     use_scheduler = False,
        #     **sc_expand_args
        # )

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)

        f1, _, results_dict = eval_mv4(test, model, gt_exps=gt_exps)

        for k, v in results_dict.items():
            print(k)
            if k not in "generated_exps" and k not in "gt_exps":
                print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))
        print('Test F1: {:.4f}'.format(f1))

        generated_exps = results_dict["generated_exps"]
        gt_exps = results_dict["gt_exps"]
        resutlt_cur = print_results(generated_exps, gt_exps)
        all_results["AUROC"] += resutlt_cur["AUROC"]
        all_results["AUPRC"] += resutlt_cur["AUPRC"]
        all_results["AUP"] += resutlt_cur["AUP"]
        all_results["AUR"] += resutlt_cur["AUR"]
    all_aupoc, all_auprc, all_aup, all_aur = all_results["AUROC"], all_results["AUPRC"], all_results["AUP"], all_results["AUR"]
    print('============================================================================================')
    print('Saliency AUROC: = {:.4f} +- {:.4f}'.format(np.mean(all_aupoc), np.std(all_aupoc) / np.sqrt(len(all_aupoc))))
    print('Saliency AUPRC: = {:.4f} +- {:.4f}'.format(np.mean(all_auprc), np.std(all_auprc) / np.sqrt(len(all_auprc))))
    print('Saliency AUP: = {:.4f} +- {:.4f}'.format(np.mean(all_aup), np.std(all_aup) / np.sqrt(len(all_aup))))
    print('Saliency AUR: = {:.4f} +- {:.4f}'.format(np.mean(all_aur), np.std(all_aur) / np.sqrt(len(all_aur))))


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

    parser.add_argument('--r', type = float, default = 0.5, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')

    args = parser.parse_args()

    main(args)

