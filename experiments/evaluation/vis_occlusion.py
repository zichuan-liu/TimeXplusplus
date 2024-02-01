import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
from sklearn import metrics
import pickle as pkl
import torch
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.ticker as ticker  
plt.rcParams.update({'font.size': 11})  


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)
pd.set_option('max_colwidth', 500)

datanames = ["pam", 'epilepsy', 'boiler']

alg_name = ["random", "dyna", "timex", "ours"]
based_path = "/TimeX/experiments/evaluation/results/"

tlist = [0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

data_dict = {
    "pam":"PAM",
        "epilepsy":"Epilepsy",
    "boiler":"Boiler"
}
for kk, dataname in enumerate(datanames):
    for alg in alg_name:
        csv_files = ['{}_{}_sp={}_occlusion_results.csv'.format(dataname, alg, i) for i in range(1,6)]
        
        if dataname=="boiler":
            csv_files = ['{}_{}_sp={}_occlusion_results.csv'.format(dataname, alg, i) for i in [1,2,3,5]]

        all_data = pd.DataFrame()
        for file in csv_files:
            df = pd.read_csv(based_path+file)
            all_data = pd.concat([all_data, df])

        auroc_means = all_data.groupby('thresh')['auroc'].mean()
        auroc_std = all_data.groupby('thresh')['auroc'].std()
        auroc_se = auroc_std / np.sqrt(len(csv_files))  # 标准误差


        # 绘制条形图
        axes[kk].plot(range(len(tlist)), auroc_means, '-o', label = alg)

        # 使用fill_between添加误差区间
        axes[kk].fill_between(range(len(tlist)), auroc_means - auroc_se, auroc_means + auroc_se, alpha=0.25)

    axes[kk].set_title(data_dict[dataname])
    axes[kk].set_xlabel('Bottom Proportion Perturbed')
    axes[kk].set_ylabel('Prediction AUROC')
    axes[kk].set_xticks(range(len(tlist)))  
    axes[kk].set_xticklabels([str(num) for num in tlist])  # Set the tick labels as strings  

    if dataname!="boiler":
        axes[kk].yaxis.set_major_locator(ticker.MultipleLocator(0.05))  
    else:
        axes[kk].yaxis.set_major_locator(ticker.MultipleLocator(0.1))  
fig.tight_layout()
h, lab = axes[0].get_legend_handles_labels()
lab = ["random", "dyna", "timex", "ours"]
lab_name = {
            "random":"Random",
            "dyna":"Dynamask",
    "timex":"Timex",
    "ours":"Ours",
}
order = [0,1,2,3]

plt.legend([h[i] for i in order], [lab_name[lab[i]] for i in order], ncol = len(lab), bbox_to_anchor=(-0.1,1.25), fontsize=12)

plt.show()

plt.savefig("./vis_results/{}.pdf".format("mean_perturbated"), bbox_inches="tight")




