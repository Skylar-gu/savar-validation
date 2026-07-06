import numpy as np, glob
from pathlib import Path
K=3
# use finecadence realisations: latent_states (8,T)
paths=sorted(glob.glob("data/realisations_finecadence/realisation_*.npz"))[:100]
Zt=[]; ZtK=[]
for p in paths:
    Z=np.load(p)["latent_states"]  # (8,T)
    Zt.append(Z[:, :-K].T); ZtK.append(Z[:, K:].T)
Zt=np.vstack(Zt); ZtK=np.vstack(ZtK)   # (M,8)
# per-mode lag-K autocorrelation and implied R^2 of AR(1)-persistence forecast
print("per-mode lag-K autocorr  ->  R^2 = rho^2 (best scalar-persistence forecast):")
for j in range(8):
    rho=np.corrcoef(Zt[:,j],ZtK[:,j])[0,1]
    print(f"  X{j}: rho(lag {K})={rho:+.3f}   R^2={rho**2:.3f}")
# multivariate optimal linear forecast Z(t+K) from Z(t): var explained
from numpy.linalg import lstsq
W,_,_,_=lstsq(np.hstack([Zt,np.ones((len(Zt),1))]),ZtK,rcond=None)
pred=np.hstack([Zt,np.ones((len(Zt),1))])@W
r2=1-((pred-ZtK)**2).sum()/((ZtK-ZtK.mean(0))**2).sum()
print(f"\nmultivariate optimal linear mode-forecast R^2 (ceiling in mode space): {r2:.3f}")
print(f"mean |rho| over modes: {np.mean([abs(np.corrcoef(Zt[:,j],ZtK[:,j])[0,1]) for j in range(8)]):.3f}")
