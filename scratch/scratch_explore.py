import numpy as np, glob
def analyze(d, label):
    paths=sorted(glob.glob(f"data/{d}/realisation_*.npz"))[:100]
    Zall=[np.load(p)["latent_states"] for p in paths]
    print(f"\n=== {label} ({d}) ===")
    print("  mode  std   lag1ρ  opt-gain(φ̂)  τ̂=-1/lnρ")
    gains=[]
    for j in range(8):
        a=np.concatenate([z[j,:-1] for z in Zall]); b=np.concatenate([z[j,1:] for z in Zall])
        rho=np.corrcoef(a,b)[0,1]
        # 1-step optimal scalar gain = cov(b,a)/var(a)
        g=np.cov(b,a)[0,1]/np.var(a)
        gains.append(g)
        tau=-1/np.log(rho) if 0<rho<1 else float('inf')
        std=np.concatenate([z[j] for z in Zall]).std()
        print(f"   X{j}  {std:.2f}  {rho:+.3f}   {g:+.3f}      {tau:.2f}")
    print(f"  gain spread (max/min): {max(gains)/min(gains):.1f}×   gains σ={np.std(gains):.3f}")
analyze("realisations_finecadence","BASELINE finecadence")
analyze("realisations_hetdynamics","VARIANT hetdynamics")
