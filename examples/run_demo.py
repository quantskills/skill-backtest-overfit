"""
Demo: contrast an OVERFIT selection (best of many noise strategies) against a
strategy with a GENUINE edge. Run:

    python examples/run_demo.py

No credentials required — fully synthetic and deterministic.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from overfit_report import build_report, render_text  # noqa: E402

rng = np.random.default_rng(42)
T, N = 1000, 200

# ---- Case A: overfit. 200 pure-noise configs; we "discover" the best one. ----
noise = rng.normal(0, 0.01, size=(T, N))
sharpes = noise.mean(0) / noise.std(0, ddof=1)
best = int(np.argmax(sharpes))
print("\n########## CASE A: best of 200 NOISE strategies (overfit) ##########")
print(render_text(build_report(noise[:, best], n_trials=N, trials_matrix=noise)))

# ---- Case B: genuine edge embedded among the same noise. --------------------
edge = noise.copy()
edge[:, best] += 0.0006  # small but real daily drift
print("\n########## CASE B: same config but with a REAL edge ##########")
print(render_text(build_report(edge[:, best], n_trials=N, trials_matrix=edge)))
