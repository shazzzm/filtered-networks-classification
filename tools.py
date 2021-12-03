import pandas as pd
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
import networkx as nx

def build_breast_cancer_dataset(desired_p):
    df = pd.read_csv("breast.csv", sep=',', index_col=0)
    df = df.T
    labels = df.index
    gene_labels = df.T.index
    # Remove the controls from the tumours
    tumour_data = []
    control_data = []

    interesting_gene_names = ["PIK3CA", "PTEN", "AKT1", "TP53", "GATA3", "CDH1", "RB1", "MLL3", "MAP3K1", "CDKN1B", "TBX3", "RUNX1", \
        "CBFB", "AFF2", "PIK3R1", "PTPN22", "PTPRD", "NF1", "SF3B1", "CCND3"]

    patient = list()

    for i,lab in enumerate(labels):
        lab_int = int(lab[-2:])
        if lab_int < 10:
            tumour_data.append(i)
            patient.append('Tumour')
        else:
            control_data.append(i)
            patient.append('Control')

    control_vals = df.iloc[control_data].values
    tumour_vals = df.iloc[tumour_data].values

    #X = df.values

    # Remove any variables with 0 variance
    v = np.var(control_vals, axis=0)
    v_2 = np.var(tumour_vals, axis=0)
    v_all = np.sort(np.concatenate((v, v_2)))
    idx_threshold = -1
    threshold = None
    num_control = []
    num_tumour = []
    for i,val in enumerate(v_all):
        num_control.append(np.count_nonzero(v > val))
        num_tumour.append(np.count_nonzero(v_2 > val))
        if np.count_nonzero(np.logical_and(v > val, v_2 > val)) >= desired_p:
            threshold = val    
            idx_threshold = i

    idx = np.logical_and(v > threshold, v_2 > threshold)

    control_vals = control_vals[:, idx]
    tumour_vals = tumour_vals[:, idx]
    gene_labels = gene_labels[idx]
    n, p = control_vals.shape

    return control_vals, tumour_vals

def build_random_correlation_matrix(p):
    C = make_spd_matrix(p)
    new_C = C.copy()
    for i in range(p):
        for j in range(p):
            new_C[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return new_C

def calculate_corr_between_graphs(G, M):
    """
    Calculates the correlation between the edge weights of a networkx graph G
    and a correlation matrix M
    """
    M_2 = nx.to_numpy_array(G)
    p = M.shape[0]
    ind = ~np.eye(p, dtype=bool)
    return np.corrcoef(M_2[ind].flatten(), M[ind].flatten())[0, 1]