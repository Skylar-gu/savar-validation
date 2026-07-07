"""Verify regenerated npz data against a content-hash manifest.

Hashes array CONTENTS (sorted keys: name|dtype|shape|bytes), not npz file
bytes — npz is a zip whose raw bytes embed timestamps and are not stable
across regeneration even when the arrays are identical.

Usage:  python3 data_gen/manifests/verify_content_hashes.py \
            data_gen/manifests/overlap02_realisations.sha256  [data_root=data]
"""
import numpy as np, hashlib, sys, os

manifest = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "data"

bad = missing = ok = 0
for line in open(manifest):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    want, rel = line.split(None, 1)
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        print(f"MISSING  {rel}"); missing += 1; continue
    d = np.load(path)
    h = hashlib.sha256()
    for k in sorted(d.files):
        a = np.ascontiguousarray(d[k])
        h.update(k.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    if h.hexdigest() == want:
        ok += 1
    else:
        print(f"MISMATCH {rel}"); bad += 1

print(f"\n{ok} ok, {bad} mismatch, {missing} missing")
sys.exit(1 if (bad or missing) else 0)
