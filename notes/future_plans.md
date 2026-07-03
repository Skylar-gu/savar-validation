Use nondisjointness after doing the contemporaneous edges experiment with subsampling: 
# overlap knob: closer centers + wider bumps
specs = [dict(center=(cy, cx), sigma=6.0, ...) for ...]   # e.g. centers 10 apart, σ=6
W, maps = build_modes(Ly, Lx, specs, normalize=True)
G = W @ W.T                       # off-diagonals = cosine overlap; target ~0.2 (mod), >0.5 (hard)
x_hat = d @ np.linalg.pinv(W).T   # NOT d @ W.T once non-orthogonal — recovery now degrades (the point)


That "can aliased contemporaneous edges even be oriented?" question is itself a worthwhile experiment.
- can already check if PCMCI+ gets the lags right 
- there isnt enough data to detect sub 6h dynamics -- could look through and include domain knowledge 

- can SAEs encode sub6h data? 