import numpy as np

def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def sqdist_1d(col):
    col = col.astype(float)
    return (col[:, None] - col[None, :]) ** 2

def loo_acc_from_d2(d2, labels):
    d2 = d2.copy()
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    return float((labels[nn] == labels).mean() * 100.0)

def greedy_select_fast(sig, classical, labels, k=12):
    labels = np.asarray(labels)
    Zc = zscore(classical.astype(float))
    d2_base = np.zeros((len(labels), len(labels)))
    for c in range(Zc.shape[1]):
        d2_base += sqdist_1d(Zc[:, c])

    Zs = zscore(sig.astype(float))
    col_d2 = [sqdist_1d(Zs[:, j]) for j in range(Zs.shape[1])]

    chosen = []
    remaining = list(range(sig.shape[1]))
    cur_d2 = d2_base.copy()
    accs = []
    for step in range(k):
        best = None
        for j in remaining:
            trial = cur_d2 + col_d2[j]
            acc = loo_acc_from_d2(trial, labels)
            if best is None or acc > best[1]:
                best = (j, acc)
        chosen.append(best[0])
        cur_d2 = cur_d2 + col_d2[best[0]]
        remaining.remove(best[0])
        accs.append(best[1])
    return chosen, accs
