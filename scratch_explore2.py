import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
D = Path("sae_data_gnn")
acts = np.load(D/"activations_full.npy"); Z = np.load(D/"Z_full.npy")
align = np.load(D/"alignment_per_mode.npy", allow_pickle=True).item()

class SAE(nn.Module):
    def __init__(s):
        super().__init__(); s.k=25
        s.encoder=nn.Linear(256,512); s.decoder=nn.Linear(512,256)
    def encode(s,x):
        pre=s.encoder(x); v,i=torch.topk(pre,s.k,dim=-1)
        a=torch.zeros_like(pre); a.scatter_(-1,i,F.relu(v)); return a

for j in [1,3,5]:
    ck=torch.load(D/f"sae_mode_{j}.pt",map_location="cpu",weights_only=False)
    sae=SAE(); sae.load_state_dict(ck["model_state"]); sae.eval()
    mean_j,std_j = ck["act_mean"], ck["act_std"]
    Aj = acts[:,j,:,:].reshape(-1,256).astype(np.float32)
    Xn = (Aj-mean_j)/std_j                       # normalized space SAE sees
    # PCA in normalized space
    Xc = Xn - Xn.mean(0)
    sub = np.random.default_rng(0).choice(len(Xc),40000,replace=False)
    U,S,Vt=np.linalg.svd(Xc[sub],full_matrices=False); var=S**2; var/=var.sum()
    PC=Vt[:3]
    # feature dominance: mean activation over data (freq*magnitude)
    with torch.no_grad():
        fa=[]
        for i in range(0,len(Xn),8192):
            fa.append(sae.encode(torch.from_numpy(Xn[i:i+8192])).numpy())
        fa=np.concatenate(fa)   # (N,512)
    dom = fa.mean(0)                 # mean activation per feature
    freq=(fa>0).mean(0)
    dec=ck["model_state"]["decoder.weight"].numpy()  # (256,512)
    cap=((PC@dec)**2).sum(0)/(np.linalg.norm(dec,axis=0)**2)
    order=np.argsort(-dom)
    top=order[:20]
    print(f"\n=== mode {j}  var top3={var[:3].sum()*100:.1f}%  bestfeat={align[j]['best_feat']} r={align[j]['max_r']:.3f}")
    print(f"  live features (freq>0.5%): {(freq>0.005).sum()} / 512")
    print(f"  top-20 dominant feats: 3D-captured frac mean {cap[top].mean():.2f} min {cap[top].min():.2f}")
    print(f"  best-aligned feat 3D-captured: {cap[align[j]['best_feat']]:.2f}, dom-rank {list(order).index(align[j]['best_feat'])}")
