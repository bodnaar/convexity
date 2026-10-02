import numpy as np
import pandas as pd

def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def loo_1nn_acc(X, labels):
    X = zscore(np.asarray(X, dtype=float))
    labels = np.asarray(labels)
    n = len(labels)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    return float((labels[nn] == labels).mean() * 100.0)

def run(name, csv_path):
    df = pd.read_csv(csv_path)
    labels = df["cls"].values
    classical_cols = ["area_ratio", "circularity", "hu1", "hu2"]
    hu17_cols = [f"hu{i}" for i in range(1, 8)]
    sig_cols = [f"sig{i}" for i in range(64)]
    sig = df[sig_cols].values
    mu = sig.mean(axis=1)

    print(f"\n=== {name} ({len(df)} shapes, {df['cls'].nunique()} classes) ===")
    feats = {
        "classical (area ratio + circ + Hu1 + Hu2)": df[classical_cols].values,
        "Hu 1-7": df[hu17_cols].values,
        "mu alone": mu.reshape(-1, 1),
        "all 64 raw signature components": sig,
        "classical + mu": np.column_stack([df[classical_cols].values, mu]),
        "classical + all 64 raw components": np.column_stack([df[classical_cols].values, sig]),
    }
    results = {}
    for label, X in feats.items():
        acc = loo_1nn_acc(X, labels)
        results[label] = acc
        print(f"  {label:45s} dims={X.shape[1]:3d}  acc={acc:6.2f}%")

    # correlation structure
    corr = np.corrcoef(sig, rowvar=False)
    iu = np.triu_indices(64, k=1)
    mean_corr_raw = corr[iu].mean()
    sig_centered = sig - mu[:, None]
    corr_c = np.corrcoef(sig_centered, rowvar=False)
    mean_corr_centered = corr_c[iu].mean()
    print(f"  mean pairwise corr of 64 raw components: {mean_corr_raw:+.3f}")
    print(f"  mean pairwise corr after removing mu:    {mean_corr_centered:+.3f}")

    corr_mu_ar = np.corrcoef(mu, df["area_ratio"].values)[0, 1]
    corr_mu_circ = np.corrcoef(mu, df["circularity"].values)[0, 1]
    print(f"  corr(mu, area_ratio) = {corr_mu_ar:+.3f}, corr(mu, circularity) = {corr_mu_circ:+.3f}")

    classes = sorted(df["cls"].unique())
    class_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_idx[c] for c in labels])
    def f_ratio(x):
        overall_mean = x.mean()
        ss_between = sum(((x[y == k].mean() - overall_mean) ** 2) * (y == k).sum() for k in range(len(classes)))
        ss_within = sum(((x[y == k] - x[y == k].mean()) ** 2).sum() for k in range(len(classes)))
        df_between = len(classes) - 1
        df_within = len(x) - len(classes)
        return (ss_between / df_between) / (ss_within / df_within)
    f_mu = f_ratio(mu)
    f_components = [f_ratio(sig[:, i]) for i in range(64)]
    print(f"  F(mu) = {f_mu:.2f}, mean F(raw components) = {np.mean(f_components):.2f}")

    return results, df

if __name__ == "__main__":
    import os
    _h = os.path.dirname(os.path.abspath(__file__))
    run("Animal2000", os.path.join(_h, "animal2000_features.csv"))
    run("SwedishLeaves", os.path.join(_h, "swedishleaves_features.csv"))

def greedy_select(sig, classical, labels, k=12):
    """Greedy forward selection of k signature components (by index into sig's
    64 columns), added on top of the classical features, scored by LOO 1NN
    accuracy on the full set -- optimistic (fit=eval) protocol. Returns (chosen
    indices, accuracy at k)."""
    labels = np.asarray(labels)
    chosen = []
    remaining = list(range(sig.shape[1]))
    best_acc = None
    accs_at_k = []
    base = classical
    for step in range(k):
        best = None
        for j in remaining:
            X = np.column_stack([base] + [sig[:, c] for c in chosen] + [sig[:, j]])
            acc = loo_1nn_acc(X, labels)
            if best is None or acc > best[1]:
                best = (j, acc)
        chosen.append(best[0])
        remaining.remove(best[0])
        accs_at_k.append(best[1])
    return chosen, accs_at_k
