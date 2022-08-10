import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import math
import networkx as nx
import os
import pandas as pd
from pathlib import Path
import operator
import matplotlib
import topcorr
from scipy.stats import spearmanr
from sklearn.covariance import ledoit_wolf
from itertools import combinations

def gini(x):
    n = x.shape[0]
    diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
    return diffs / (2 * n**2 * x.mean())

def calculate_corr_between_graphs(G, M):
    """
    Calculates the correlation between the edge weights of a networkx graph G
    and a correlation matrix M
    """
    M_2 = nx.to_numpy_array(G)
    p = M.shape[0]
    ind = ~np.eye(p, dtype=bool)
    return np.corrcoef(M_2[ind].flatten(), M[ind].flatten())[0, 1]

def calculate_graph_diff(M_1, M_2, is_nx_graph = False):
    """
    Calculates the normalized weighted difference between two graphs
    """
    nodes = np.arange(p).tolist()

    if is_nx_graph:
        M_1 = nx.to_numpy_array(M_1, nodes)
        M_2 = nx.to_numpy_array(M_2, nodes)

    diff = np.sum(np.abs(M_1 - M_2).flatten())
    return( diff / (np.sum(np.abs(M_1).flatten()) * np.sum(np.abs((M_2)).flatten())))

# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.rcParams.update({'figure.max_open_warning': 0})

# Set the country you desire to analyze
country = "DE"
if country == "DE":
    df = pd.read_csv("DAX30.csv", index_col=0)        
    window_size = 252 * 2
elif country == "UK":
    df = pd.read_csv("FTSE100.csv", index_col=0)
    window_size = 252 * 2
elif country == "US":
    df = pd.read_csv("S&P500.csv", index_col=0)
    window_size = 252 * 2
elif country == "CH":
    df = pd.read_csv("SSE50.csv", index_col=0)
    window_size = 252 * 2
elif country == "IN":
    df = pd.read_csv("NIFTY50.csv", index_col=0)
    window_size = 252 * 2

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sector_set = sorted(set(company_sectors))

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2.index = pd.to_datetime(df_2.index)
volatility = []
df_2 = np.log(df_2) - np.log(df_2.shift(1))

X = df_2.values[1:, :]

slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df_2.index[(x+1)*slide_size+window_size])
dt = pd.to_datetime(dates)
dt_2 = dt[1:]

degree_corr_mst = []
degree_corr_pmfg = []
degree_corr_tmfg = []

edge_corr_mst = []
edge_corr_pmfg = []
edge_corr_tmfg = []


for x in range(no_runs):
    print("Run %s" % x)

    X_new = X[x*slide_size:x*slide_size+window_size, :]
    C = np.corrcoef(X_new.T)
    C = np.abs(C)
    
    # Get the correlation coefficients
    ind = np.triu_indices(p, 1)
    corr_vals = C[ind].flatten()

    C_lw = np.abs(ledoit_wolf(X_new)[0])

    nodes = np.arange(p).tolist()
    G_mst_1 = topcorr.mst(C)
    G_pmfg_1 = topcorr.pmfg(C)
    G_tmfg_1 = topcorr.tmfg(C)


    full_degree = C.sum(axis=0).flatten().tolist()
    mst_degree =  list([x[1] for x in G_mst_1.degree(nodes, weight='weight')])
    pmfg_degree =  list([x[1] for x in G_pmfg_1.degree(nodes, weight='weight')])
    tmfg_degree =  list([x[1] for x in G_tmfg_1.degree(nodes, weight='weight')])
    
    degree_corr_mst.append(np.corrcoef(full_degree, mst_degree)[0, 1])
    degree_corr_pmfg.append(np.corrcoef(full_degree, pmfg_degree)[0, 1])
    degree_corr_tmfg.append(np.corrcoef(full_degree, tmfg_degree)[0, 1])

    edge_corr_mst.append(calculate_corr_between_graphs(G_mst_1, C))
    edge_corr_pmfg.append(calculate_corr_between_graphs(G_pmfg_1, C))
    edge_corr_tmfg.append(calculate_corr_between_graphs(G_tmfg_1, C))

    volatility.append(np.var(X_new, axis=0).mean())

plt.figure()
degree_corr_mst = pd.Series(degree_corr_mst, index = dates)
degree_corr_mst.name = "MST"
degree_corr_mst.plot()
degree_corr_pmfg = pd.Series(degree_corr_pmfg, index = dates)
degree_corr_pmfg.name = "PMFG"
degree_corr_pmfg.plot()
degree_corr_tmfg = pd.Series(degree_corr_tmfg, index = dates)
degree_corr_tmfg.name = "TMFG"
degree_corr_tmfg.plot()
plt.ylim([0.25, 1])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("degree_corr_over_time_%s.png" % country)

plt.figure()
edge_corr_mst = pd.Series(edge_corr_mst, index = dates)
edge_corr_mst.name = "MST"
edge_corr_mst.plot()
edge_corr_pmfg = pd.Series(edge_corr_pmfg, index = dates)
edge_corr_pmfg.name = "PMFG"
edge_corr_pmfg.plot()
edge_corr_tmfg = pd.Series(edge_corr_tmfg, index = dates)
edge_corr_tmfg.name = "TMFG"
edge_corr_tmfg.plot()
plt.ylim([0, 0.75])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("edge_corr_over_time_%s.png" % country)

print("Degree Corr")
print("MST")
print(spearmanr(degree_corr_mst, volatility))
print("PMFG")
print(spearmanr(degree_corr_pmfg, volatility))
print("TMFG")
print(spearmanr(degree_corr_tmfg, volatility))

print("Edge Corr")
print("MST")
print(spearmanr(edge_corr_mst, volatility))
print("PMFG")
print(spearmanr(edge_corr_pmfg, volatility))
print("TMFG")
print(spearmanr(edge_corr_tmfg, volatility))

