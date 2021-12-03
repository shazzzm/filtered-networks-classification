import topcorr
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
import scipy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import tools
import pandas as pd

# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

country = 'DE'
window_type = 'min' # window type = max, min or median volatility

if country == 'US':
    ps = np.arange(10, 210, 10)
elif country == 'UK':
    ps = np.arange(10, 70, 5)
elif country == 'DE':
    ps = np.arange(10, 23, 1)

max_p = max(ps)


if country == "US":
    df = pd.read_csv("S&P500.csv", index_col=0)
elif country == "UK":
    df = pd.read_csv("FTSE100.csv", index_col=0)
elif country == "DE":
    df = pd.read_csv("DAX30.csv", index_col=0)

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2.index = pd.to_datetime(df_2.index)
df_2 = np.log(df_2) - np.log(df_2.shift(1))


if window_type == 'med':
    mean_std = df_2.rolling(512).std().mean(axis=1)
    argmed = np.argsort(mean_std)[len(mean_std)//2]

    X_min = df_2.iloc[argmed-252:argmed+252, :].values
    X_max = df_2.iloc[argmed+252:argmed+756, :].values
else:
    mean_std = df_2.rolling(512).std().mean(axis=1)
    min_std = mean_std.argmin()
    max_std = mean_std.argmax()

    X_min = df_2.iloc[min_std-252:min_std+252, :].values
    X_max = df_2.iloc[max_std-252:max_std+252, :].values

if window_type == 'max':
    X_stocks = X_max
elif window_type == 'min':
    X_stocks = X_min 
elif window_type == 'med':
        X_stocks = X_min


num_runs = 20

full_degree = []
mst_degree = []
pmfg_degree = []
tmfg_degree = []
ps_label = []

corrs_mean_mst = []
corrs_mean_pmfg = []
corrs_mean_tmfg = []

corrs_stdev_mst = []
corrs_stdev_pmfg = []
corrs_stdev_tmfg = []

edge_corrs_mean_mst = []
edge_corrs_mean_pmfg = []
edge_corrs_mean_tmfg = []

edge_corrs_stdev_mst = []
edge_corrs_stdev_pmfg = []
edge_corrs_stdev_tmfg = []


for p in ps:
    print(p)

    tmp_corr_mst = []
    tmp_corr_pmfg = []
    tmp_corr_tmfg = []

    tmp_edge_corr_mst = []
    tmp_edge_corr_pmfg = []
    tmp_edge_corr_tmfg = []

    for k in range(num_runs):
        ind = np.random.choice(max_p, size=p)
        C = np.corrcoef(X_stocks[:, ind].T)
        C = np.abs(C)

        #C = np.abs(C)
        nodes = np.arange(p).tolist()
        G_mst_1 = topcorr.mst(C)
        G_pmfg_1 = topcorr.pmfg(C)
        G_tmfg_1 = topcorr.tmfg(C)

        full_degree += C.sum(axis=0).flatten().tolist()
        mst_degree +=  list([x[1] for x in G_mst_1.degree(nodes, weight='weight')])
        pmfg_degree +=  list([x[1] for x in G_pmfg_1.degree(nodes, weight='weight')])
        tmfg_degree +=  list([x[1] for x in G_tmfg_1.degree(nodes, weight='weight')])
        ps_label += [p] * p

        tmp_corr_mst.append(np.corrcoef(full_degree, mst_degree)[0, 1])
        tmp_corr_pmfg.append(np.corrcoef(full_degree, pmfg_degree)[0, 1])
        tmp_corr_tmfg.append(np.corrcoef(full_degree, tmfg_degree)[0, 1])
        
        tmp_corr_mst.append(np.corrcoef(full_degree, mst_degree)[0, 1])
        tmp_corr_pmfg.append(np.corrcoef(full_degree, pmfg_degree)[0, 1])
        tmp_corr_tmfg.append(np.corrcoef(full_degree, tmfg_degree)[0, 1])

        tmp_edge_corr_mst.append(tools.calculate_corr_between_graphs(G_mst_1, C))
        tmp_edge_corr_pmfg.append(tools.calculate_corr_between_graphs(G_pmfg_1, C))
        tmp_edge_corr_tmfg.append(tools.calculate_corr_between_graphs(G_tmfg_1, C))

    corrs_mean_mst.append(np.mean(tmp_corr_mst))
    corrs_mean_pmfg.append(np.mean(tmp_corr_pmfg))
    corrs_mean_tmfg.append(np.mean(tmp_corr_tmfg))

    corrs_stdev_mst.append(np.std(tmp_corr_mst))
    corrs_stdev_pmfg.append(np.std(tmp_corr_pmfg))
    corrs_stdev_tmfg.append(np.std(tmp_corr_tmfg))

    edge_corrs_mean_mst.append(np.mean(tmp_edge_corr_mst))
    edge_corrs_mean_pmfg.append(np.mean(tmp_edge_corr_pmfg))
    edge_corrs_mean_tmfg.append(np.mean(tmp_edge_corr_tmfg))

    edge_corrs_stdev_mst.append(np.std(tmp_edge_corr_mst))
    edge_corrs_stdev_pmfg.append(np.std(tmp_edge_corr_pmfg))
    edge_corrs_stdev_tmfg.append(np.std(tmp_edge_corr_tmfg))

ylims = [0, 20]
xlims = [-20, 70]

plt.figure()
plt.errorbar(ps, corrs_mean_mst, yerr = corrs_stdev_mst, label='MST')
plt.errorbar(ps, corrs_mean_pmfg, yerr = corrs_stdev_pmfg, label='PMFG')
plt.errorbar(ps, corrs_mean_tmfg, yerr = corrs_stdev_tmfg, label='TMFG')
plt.xlabel("$p$")
plt.ylabel("Degree Correlation")
plt.legend(loc='upper right')
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("degree_corr_%s_%s.png" % (country, window_type))

plt.figure()
plt.errorbar(ps, edge_corrs_mean_mst, yerr = edge_corrs_stdev_mst, label='MST')
plt.errorbar(ps, edge_corrs_mean_pmfg, yerr = edge_corrs_stdev_pmfg, label='PMFG')
plt.errorbar(ps, edge_corrs_mean_tmfg, yerr = edge_corrs_stdev_tmfg, label='TMFG')
plt.xlabel("$p$")
plt.ylabel("Edge Correlation")
plt.legend(loc='upper right')
plt.ylim([0, 1])

if country == "UK":
    plt.xlim([8, 65])
plt.tight_layout()
plt.savefig("edge_corr_%s_%s.png" % (country, window_type))
plt.close('all')