import numpy as np, glob
paths=sorted(glob.glob("data/realisations_finecadence/realisation_*.npz"))[:100]
Zall=[np.load(p)["latent_states"] for p in paths]   # each (8,T)
Z=np.concatenate([z for z in Zall],axis=1)          # (8, 100*T)

print("per-mode  mean      std      var")
for j in range(8):
    print(f"  X{j}:  {Z[j].mean():+.4f}   {Z[j].std():.4f}   {Z[j].var():.4f}")
print(f"  overall std across modes: min {Z.std(1).min():.3f} max {Z.std(1).max():.3f} "
      f"ratio {Z.std(1).max()/Z.std(1).min():.2f}x")

# autocorrelation at lags 1,2,3 (within each realisation, then pooled)
print("\nper-mode lag-1 / lag-2 / lag-3 autocorrelation (fine steps):")
for j in range(8):
    r=[]
    for lag in (1,2,3):
        a=np.concatenate([z[j,:-lag] for z in Zall]); b=np.concatenate([z[j,lag:] for z in Zall])
        r.append(np.corrcoef(a,b)[0,1])
    print(f"  X{j}:  {r[0]:+.3f}  {r[1]:+.3f}  {r[2]:+.3f}")

# CORRECT ceiling: 1-step-ahead optimal linear forecast of Z(t+1)
# (a) from Z(t) only ; (b) from [Z(t-2),Z(t-1),Z(t)] (3 frames, like the GNN)
def stack(hist):
    Xs=[]; Ys=[]
    for z in Zall:
        T=z.shape[1]
        Y=z[:, hist:].T                                   # (T-hist, 8)  targets Z(t)
        cols=[z[:, hist-1-k:T-1-k].T for k in range(hist)] # k=0 is most recent
        Xs.append(np.hstack(cols)); Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)
for hist,label in [(1,"Z(t) -> Z(t+1)"),(3,"[Z(t-2..t)] -> Z(t+1)")]:
    X,Y=stack(hist)
    from numpy.linalg import lstsq
    Xa=np.hstack([X,np.ones((len(X),1))])
    W,_,_,_=lstsq(Xa,Y,rcond=None); P=Xa@W
    r2=1-((P-Y)**2).sum()/((Y-Y.mean(0))**2).sum()
    print(f"\n1-step optimal linear mode-forecast R^2  [{label}]: {r2:.3f}")
