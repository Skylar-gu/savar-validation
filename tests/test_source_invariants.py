"""Regression guards on source/structure — pure text checks, no deps, always run.

These preserve the fixes and conventions established during development so they
can't silently regress:
  - extract_activations.py infers T from data (was hardcoded 500 → broke T=2920)
  - scripts resolve savar/instantiate_model relative to the repo root (post-reorg)
  - extract imports cnn_forecaster from train/ (post-reorg)
  - .gitignore keeps large/vendored paths out of the repo
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel):
    return (ROOT / rel).read_text()


def test_extract_infers_T_dynamically():
    s = read("sae/extract_activations.py")
    assert "T_TOTAL" in s              # length read from data
    assert "reshape(500" not in s      # the old hardcoded T must be gone
    assert "T_eff   = 500 - K" not in s


def test_generators_resolve_savar_from_root():
    needle = "dirname(os.path.dirname(os.path.abspath(__file__)))"
    for f in ("data_gen/instantiate_model.py",
              "data_gen/generate_dataset.py",
              "data_gen/generate_dy005.py",
              "data_gen/generate_diurnal.py"):
        assert needle in read(f), f"{f} must resolve savar relative to repo root"


def test_extract_imports_cnn_from_train():
    s = read("sae/extract_activations.py")
    assert 'parent.parent / "train"' in s
    assert "from cnn_forecaster import" in s


def test_gitignore_excludes_large_and_vendored():
    g = read(".gitignore")
    for pat in ("savar/", "data/", "sae_data", "checkpoints", "__pycache__", "*.log"):
        assert pat in g, f".gitignore must contain {pat!r}"


def test_sae_decoder_resample_uses_column_indexing():
    # decoder weight is (input_dim, n_features); dead features are columns.
    s = read("sae/train_sae_per_mode.py")
    assert 'dec_state["exp_avg"][:, dead_idx]' in s
