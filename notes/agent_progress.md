# Agent session progress log — 2026-07-03

Branch: `agent/session-2026-07-03` (off `phase5-7-8-analysis`).
Plan: `notes/agent_session_plan.md`. Results: `results/*.npy` + `notes/session_results.md`.

---

- **[start] Block A** — VPD (M4 config) on `checkpoints_hetdynamics_eqvar/best.pt`.
  Config: `vpd/config_gnn_vpd_m4_eqvar.yaml` (byte-identical loss/CI block to the
  p-0625f7dd finecadence run; only checkpoint_path + split_dir changed). Launching
  in background on the L40S.

- **[resume 04:26] Session resumed by follow-up agent.** VPD eqvar run in flight
  (`vpd_out/runs/p-6b9ba3ba`, ~6.4 it/s, ETA ~04:32). Verified
  `vpd/analyze_redundancy.py` reproduces p-0625f7dd baselines exactly
  (layer0: PR=1.14, med r=0.787, blobCV=0.050). Block C metric script
  (`sae/eval_sae_metrics.py`) written.
- **[plan change 04:35]** User promoted KAN-SAE from stretch to **Block C2**
  (immediately after C): SAE_ARCH=kan flag on train_sae_mixed.py, spline gates on
  encoder pre-acts only, TopK + linear decoder unchanged, same 3 seeds, paired
  Hungarian scoring, deliverable results/kan_sae_bakeoff.npy. D/F reruns only if
  it wins beyond seed error bars. New order: A, B, C, C2, D, E, F, G, H.
- [04:40] pip install lingam==1.12.2, pydmd, python-louvain into main env (plan-sanctioned, Blocks D/E/H).

- **[04:45] Block A primary DONE — success criterion NOT met.** Run
  `vpd_out/runs/p-6b9ba3ba` (5000 steps, 13 min). Redundancy vs p-0625f7dd
  baseline (layer0.mlp0): PR 1.08 (base 1.14), med pairwise r 0.813 (base 0.787),
  blobCV 0.038 (base 0.050). All 8 modules PR 1.08-1.54 — far below the >3/64
  success bar. Mild late-layer movement: layers_2_mlp_2 PR 1.29->1.54,
  #mode-preferential 8->12; layers_3_mlp_2 #pref 7->14, sp(phi,ent) flips +0.34.
  Verdict so far: timescale-heterogeneous data did NOT differentiate components
  -> per plan, launched the SECONDARY config test (C=16, ImportanceMinimality
  x30) before concluding method-vs-data. Results: results/vpd_eqvar_redundancy.npy.
- **[04:46] Block A secondary launched** (`vpd/config_gnn_vpd_m4_eqvar_c16.yaml`,
  log scratchpad/vpd_eqvar_c16.log). Meanwhile running Block B on CPU.
- **[05:00 resume #2]** Session resumed again. Found Block A secondary (C=16,
  IM x30) DEAD — no process, GPU idle, no run dir, no log (died with the previous
  session before writing anything). Relaunched detached (pid 701665,
  log scratchpad/vpd_eqvar_c16.log, ~10.8 it/s, ETA ~8 min). Resuming Block B on
  CPU meanwhile.
- **[05:02 resume #3]** Session resumed. VPD c16 secondary still training
  (p-? run dir pending, ~36% done, slowed to ~2.6 it/s by GPU sharing).
  Launched Block B (`pcmci/impulse_response_gnn.py`, pid 702827,
  log scratchpad/block_b_impulse.log). Next: Block C evals on existing SAE
  artifacts while both run.
- **[05:15] Block B DONE.** `pcmci/impulse_response_gnn.py`: F1=0.500
  (P=0.625/R=0.417) vs PCMCI-on-Z 0.823 — success bar not met, but all 5
  detected edges lag-correct; misses concentrated on slow-mode effects.
  Sensitivity run (24 steps, 480 windows): identical edge sets -> robust, not a
  budget artifact; e-folding fit window-dependent (rollout damping uniform
  beyond ~10 steps). results/impulse_response{,_s24}.npy. Committing Block B.
- **[05:25] Block A secondary DONE -> Block A COMPLETE.** C=16 IM x30 run
  `out/runs/p-d67e2ebb` (note: out/ root, not vpd_out/). PR still 1.01-1.23/16
  across all 8 modules -> config amplification does not fix collapse. But
  layers_3_mlp_2 blobCV 0.851 (baseline 0.04), 12/16 mode-pref, dominant modes
  piled on X6/X7, sp(phi,ent)=+0.50: the single shared gate concentrates on
  slow-mode blobs. VERDICT: data-problem hypothesis NOT confirmed; collapse is
  method/objective-level. results/vpd_eqvar_redundancy_c16.npy. Committing A.
- **[05:35] Block C DONE.** Metric suite on all artifacts + 3-seed eqvar
  retrain: MCC finecadence 0.341 < hetdynamics 0.421 < eqvar 0.463 (permode);
  eqvar mixed seeds 0.4064+-0.0036. Hungarian matching confirms (not deflates)
  old best-|r| story; uniqueness ~0 = mode-identity failure sharpened.
  results/sae_metrics_suite.npy.
- **[05:35] Block C2 DONE.** KAN-SAE bake-off: paired dMCC=-0.0008+-0.0068 ->
  TIE; gate says NO propagation, TopK stays primary, no D/F reruns.
  results/kan_sae_bakeoff.npy.
