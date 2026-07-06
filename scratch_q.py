import numpy as np, glob
for d,label in [("realisations_hetdynamics","VARIANT (as-is)"),
                ("realisations_hetdynamics_eqvar","VARIANT (eqvar)")]:
    paths=sorted(glob.glob(f"data/{d}/realisation_*.npz"))[:100]
    Zall=[np.load(p)["latent_states"] for p in paths]
    stds=[]; gains=[]
    for j in range(8):
        a=np.concatenate([z[j,:-1] for z in Zall]); b=np.concatenate([z[j,1:] for z in Zall])
        stds.append(np.concatenate([z[j] for z in Zall]).std())
        gains.append(np.cov(b,a)[0,1]/np.var(a))
    stds=np.array(stds); gains=np.array(gains)
    print(f"\n{label}")
    print(f"  std per mode: {[round(s,2) for s in stds]}  spread {stds.max()/stds.min():.2f}×")
    print(f"  gain per mode:{[round(g,2) for g in gains]}  spread {gains.max()/gains.min():.1f}×")
