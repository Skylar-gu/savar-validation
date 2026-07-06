import numpy as np, torch, glob, sys
from pathlib import Path
sys.path.insert(0,"train")
from gnn_forecaster import MeshGNN, MultiRealisationDataset, K, run_epoch, DEVICE
from torch.utils.data import DataLoader

# load val windows the SAME way the GNN sees them
val_ds = MultiRealisationDataset("data/splits_finecadence/val", K)
X = torch.stack([val_ds[i][0] for i in range(len(val_ds))])  # (N,K,50,50)
Y = torch.stack([val_ds[i][1] for i in range(len(val_ds))]).squeeze(1)  # (N,50,50)
Xf, Yf = X.numpy(), Y.numpy()
N = len(Yf); print("val windows:", N, " frame std(grand):", Yf.std())

def rmse(pred): return float(np.sqrt(((pred-Yf)**2).mean()))
def corr(pred):
    p=pred.reshape(N,-1); t=Yf.reshape(N,-1)
    p=p-p.mean(1,keepdims=True); t=t-t.mean(1,keepdims=True)
    return float((( (p*t).sum(1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(t,axis=1)+1e-8) )).mean())

# --- baselines ---
# 1. grand mean (single scalar over all val targets)
gm = Yf.mean()
b_grand = np.full_like(Yf, gm)
# 2. per-node climatology (per-pixel time mean, computed per-realisation to be fair to "mean field")
#    approximate global per-pixel mean over all val targets
pix_mean = Yf.mean(0)                      # (50,50)
b_pix = np.broadcast_to(pix_mean, Yf.shape)
# 3. persistence: predict = last input frame
b_pers = Xf[:, -1]                         # (N,50,50)
# 4. "mean by mode": reconstruct using true modes but only their MEAN amplitude (=0) -> same as climatology field
#    contrast: reconstruct using true CURRENT mode amplitudes (an oracle that tracks modes)

print(f"\n{'baseline':<28}{'RMSE':>8}{'corr':>8}")
for name,p in [("predict grand mean", b_grand),
               ("per-pixel climatology", b_pix),
               ("persistence (copy last frame)", b_pers)]:
    print(f"{name:<28}{rmse(p):>8.4f}{corr(p):>8.4f}")

# --- GNN ---
model = MeshGNN(ny=50,nx=50,k=K).to(DEVICE)
ck = torch.load("checkpoints_finecadence/best.pt", map_location=DEVICE)
model.load_state_dict(ck["model_state"]); model.eval()
preds=[]
with torch.no_grad():
    for i in range(0,N,64):
        preds.append(model(X[i:i+64].to(DEVICE)).squeeze(1).cpu().numpy())
gnn=np.concatenate(preds)
print(f"{'GNN forecaster':<28}{rmse(gnn):>8.4f}{corr(gnn):>8.4f}")

# --- how much of GNN skill is just persistence? correlation of GNN pred with persistence ---
def frac_var_explained(a):  # 1 - MSE/var of predicting grand mean
    return 1 - ((a-Yf)**2).mean()/((Yf-gm)**2).mean()
print("\nR^2 (vs grand mean baseline):")
for name,p in [("persistence",b_pers),("GNN",gnn),("per-pixel clim",b_pix)]:
    print(f"  {name:<16}{frac_var_explained(p):+.4f}")
# GNN vs persistence: does GNN beat copy-last-frame?
print(f"\nGNN pred vs persistence pred, mean corr: {corr(b_pers)  :.4f} (pers) vs {corr(gnn):.4f} (gnn)")
# correlation between GNN prediction and simply the last frame
pp=gnn.reshape(N,-1); ll=b_pers.reshape(N,-1)
pp=pp-pp.mean(1,keepdims=True); ll=ll-ll.mean(1,keepdims=True)
c=((pp*ll).sum(1)/(np.linalg.norm(pp,axis=1)*np.linalg.norm(ll,axis=1)+1e-8)).mean()
print(f"corr(GNN prediction, last input frame): {c:.4f}")
