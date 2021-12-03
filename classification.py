import topcorr
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
import karateclub
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
import scipy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import tools

# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

def embed_correlation_matrices(corr, k):
    """
    Embeds a correlation matrix using the graph laplacian
    """

    n,m = corr.shape
    diags = corr.sum(axis=1).flatten()
    D = np.diag(diags)
    L = D - corr
    with scipy.errstate(divide='ignore'):
       diags_sqrt = 1.0/np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = np.diag(diags_sqrt)
    L = DH.dot(L.dot(DH))

    if m <= k:
        embedding = scipy.sparse.linalg.eigsh(L, k=m-1, which='LM',
                            ncv=10*k, return_eigenvectors=False)

        shape_diff = k - embedding.shape[0] - 1
        embedding = np.pad(embedding, (1, shape_diff), 'constant', constant_values=0)
    else:
        embedding = scipy.sparse.linalg.eigsh(L, k=k, which='LM',
                            ncv=10*k, return_eigenvectors=False)
    return embedding

def create_filtered_correlation_matrix(corr, method):
    return method(corr)

def run_classification(train_index, test_index, Z, y, rf_scores):
    X_train, X_test = Z[train_index, :], Z[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_scores.append(accuracy_score(y_test, y_pred))

def run_filtration_set_random(C_1, C_2):
    X = np.random.multivariate_normal(np.zeros(p), C_1, n)
    corr_1 = np.abs(np.corrcoef(X.T))
    G_mst_1 = topcorr.mst(corr_1)
    G_pmfg_1 = topcorr.pmfg(corr_1)
    G_tmfg_1 = topcorr.tmfg(corr_1)

    X = np.random.multivariate_normal(np.zeros(p), C_2, n)
    corr_2 = np.abs(np.corrcoef(X.T))
    G_mst_2 = topcorr.mst(corr_2)
    G_pmfg_2 = topcorr.pmfg(corr_2)
    G_tmfg_2 = topcorr.tmfg(corr_2)
    return (corr_1, G_mst_1, G_pmfg_1, G_tmfg_1, corr_2, G_mst_2, G_pmfg_2, G_tmfg_2)

def run_filtration_set(X_1, X_2, desired_p):
    ind = np.random.choice(X_1.shape[1], size=desired_p, replace=False)

    corr_1 = np.abs(np.corrcoef(X_1[:, ind].T))
    G_mst_1 = topcorr.mst(corr_1)
    G_pmfg_1 = topcorr.pmfg(corr_1)
    G_tmfg_1 = topcorr.tmfg(corr_1)

    corr_2 = np.abs(np.corrcoef(X_2[:, ind].T))
    G_mst_2 = topcorr.mst(corr_2)
    G_pmfg_2 = topcorr.pmfg(corr_2)
    G_tmfg_2 = topcorr.tmfg(corr_2)
    return (corr_1, G_mst_1, G_pmfg_1, G_tmfg_1, corr_2, G_mst_2, G_pmfg_2, G_tmfg_2)

# which country the stock dataset comes from
country = "US"

# which graph embedding method to use (usually this is SF)
method = karateclub.SF
method_str = 'SF'
stocks_med = False

# p will need to be changed depending on the dataset
p = 150

ns = np.arange(10, 210, 10)
k = 20
num_corr = 50

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

# This means we classify two consecutive windows of stock data of median
# risk, rather than taking the minimum and maximum risk. It should be
# a harder problem
if stocks_med:
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

full_mean = []
full_stdev = []

mst_mean = []
mst_stdev = []

pmfg_mean = []
pmfg_stdev = []

tmfg_mean = []
tmfg_stdev = []

for n in ns:
    print(n)
    ind_min = np.random.choice(X_min.shape[0], size=n, replace=False)
    ind_max = np.random.choice(X_max.shape[0], size=n, replace=False)
    X_1 = X_min[ind_min, :]
    X_2 = X_max[ind_max, :]

    #assert(np.all(C_1 > 0))
    #assert(np.all(C_2 > 0))

    C_1_mats = []
    C_2_mats = []

    C_1_msts = []
    C_2_msts = []

    C_1_pmfgs = []
    C_2_pmfgs = []

    C_1_tmfgs = []
    C_2_tmfgs = []

    C_1_tmfgs_2 = []
    C_2_tmfgs_2 = []

    methods = [topcorr.mst, topcorr.pmfg, topcorr.tmfg]

    results = Parallel(n_jobs=4)(delayed(run_filtration_set)(X_1, X_2, p) for i in range(50))

    # Disentangle the results list
    for res in results:
        C_1_mats.append(res[0])
        C_1_msts.append(res[1])
        C_1_pmfgs.append(res[2])
        C_1_tmfgs.append(res[3])
        C_2_mats.append(res[4])
        C_2_msts.append(res[5])
        C_2_pmfgs.append(res[6])
        C_2_tmfgs.append(res[7])


    y = np.array([0] * num_corr + [1] * num_corr)
    Cs = C_1_mats + C_2_mats
    C_msts = C_1_msts + C_2_msts
    C_pmfgs = C_1_pmfgs + C_2_pmfgs
    C_tmfgs = C_1_tmfgs + C_2_tmfgs

    Z = np.zeros((num_corr *2, k))

    for i,mat in enumerate(Cs):
        Z[i, :] = embed_correlation_matrices(mat, k)

    emb = method(k)
    emb.fit(C_msts)
    Z_mst = emb.get_embedding()

    emb = method(k)
    emb.fit(C_pmfgs)
    Z_pmfg = emb.get_embedding()

    emb = method(k)
    emb.fit(C_tmfgs)
    Z_tmfg = emb.get_embedding()

    kf = StratifiedKFold(10)

    rf_scores = []
    rf_scores_mst = []
    rf_scores_pmfg = []
    rf_scores_tmfg = []

    i = 0
    for train_index, test_index in kf.split(Z, y):
        i += 1
        run_classification(train_index, test_index, Z, y, rf_scores)
        run_classification(train_index, test_index, Z_mst, y, rf_scores_mst)
        run_classification(train_index, test_index, Z_pmfg, y, rf_scores_pmfg)
        run_classification(train_index, test_index, Z_tmfg, y, rf_scores_tmfg)


    full_mean.append(np.mean(rf_scores))
    full_stdev.append(np.std(rf_scores))

    mst_mean.append(np.mean(rf_scores_mst))
    mst_stdev.append(np.std(rf_scores_mst))

    pmfg_mean.append(np.mean(rf_scores_pmfg))
    pmfg_stdev.append(np.std(rf_scores_pmfg))

    tmfg_mean.append(np.mean(rf_scores_tmfg))
    tmfg_stdev.append(np.std(rf_scores_tmfg))

plt.figure()
plt.errorbar(ns, full_mean, yerr=full_stdev, label='Full')
plt.errorbar(ns, mst_mean, yerr=mst_stdev, label='MST')
plt.errorbar(ns, pmfg_mean, yerr=pmfg_stdev, label='PMFG')
plt.errorbar(ns, tmfg_mean, yerr=tmfg_stdev, label='TMFG')
plt.ylabel("Mean Accuracy")
plt.xlabel("n")
plt.legend(loc='lower right')
plt.ylim([0, 1.1])
plt.tight_layout()
if stocks_med:
    plt.savefig("classification_%s_%s_%s_med.png" % (p, method_str, country))
else:
    plt.savefig("classification_%s_%s_%s.png" % (p, method_str, country))

plt.show()