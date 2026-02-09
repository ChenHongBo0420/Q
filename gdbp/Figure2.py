"""
gdbp/Figure2.py

Figure 2 focus: "where it fails" / "upper bound collapse" WITHOUT sweeping switch_symbols.

Core idea
---------
Fix the train->track switch point (default 200000) to avoid introducing a new instability
source. For each checkpoint θ_k:
  - run ONE forward equalization to obtain the whole post-DSP sequence
  - evaluate Q on multiple time windows (same schedule, same switch)
The distribution over windows gives:
  - worst-case:   Q_min(θ_k)
  - robust bound: Q_0.1(θ_k)  (10th percentile over windows)

This turns "boundary" into a *time-window robustness boundary* (defendable) instead of a
"schedule sweep boundary" (confounded by timeline changes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

import matplotlib.pyplot as plt

import jax
from commplax.module import core
from commplax import util, comm2

# ---- import baseline/solution modules ----
try:
    from . import Problem as Problem  # baseline (MSE)
except Exception:
    import Problem as Problem  # type: ignore

try:
    from . import Solution as Solution  # ours (quotient-space)
except Exception:
    import Solution as Solution  # type: ignore


Array = Any
FrozenDict = Any


# =============================================================================
# Optional wrappers: keep your Colab minimal
# =============================================================================

def model_init(*args, variant: str = "problem", **kwargs):
    """
    variant:
      - "problem"  -> Problem.model_init
      - "solution" -> Solution.model_init
    """
    if variant.lower() in ["problem", "baseline", "mse"]:
        return Problem.model_init(*args, **kwargs)
    if variant.lower() in ["solution", "ours", "snr", "qspace"]:
        return Solution.model_init(*args, **kwargs)
    raise ValueError(f"Unknown variant={variant}")

def train(*args, variant: str = "problem", **kwargs):
    if variant.lower() in ["problem", "baseline", "mse"]:
        return Problem.train(*args, **kwargs)
    if variant.lower() in ["solution", "ours", "snr", "qspace"]:
        return Solution.train(*args, **kwargs)
    raise ValueError(f"Unknown variant={variant}")

def test(*args, variant: str = "problem", **kwargs):
    if variant.lower() in ["problem", "baseline", "mse"]:
        return Problem.test(*args, **kwargs)
    if variant.lower() in ["solution", "ours", "snr", "qspace"]:
        return Solution.test(*args, **kwargs)
    raise ValueError(f"Unknown variant={variant}")


# =============================================================================
# Windows + Q extraction
# =============================================================================

def _to_abs_stop(n: int, stop: int) -> int:
    return (n + stop) if stop < 0 else stop

def make_eval_windows(
    total_len_hint: int,
    eval_start: int = 300_000,
    eval_stop: int = -20_000,
    win_len: int = 20_000,
    stride: Optional[int] = None,
    n_windows: Optional[int] = 8,
    center: bool = True,
) -> List[Tuple[int, int]]:
    """
    Build list of (start, stop) windows in equalized index space.
    Use the SAME windows for baseline vs ours.
    """
    n = int(total_len_hint)
    s0 = int(eval_start)
    s1 = _to_abs_stop(n, int(eval_stop))
    s1 = max(s1, s0 + win_len)
    region = s1 - s0
    if region < win_len:
        return [(s0, s0 + win_len)]

    if stride is None:
        stride = win_len

    starts = list(range(s0, s1 - win_len + 1, int(stride)))
    if not starts:
        starts = [s0]

    if n_windows is None or n_windows >= len(starts):
        sel = starts
    else:
        if center:
            idx = np.linspace(0, len(starts) - 1, n_windows)
            idx = np.round(idx).astype(int).tolist()
            sel = [starts[i] for i in idx]
        else:
            sel = starts[:n_windows]

    windows = [(int(st), int(st + win_len)) for st in sel]
    windows = sorted(list(dict.fromkeys(windows)))
    return windows

def _extract_q_ber(res: Dict[str, Any]) -> Tuple[float, float]:
    """
    Robustly extract Q(dB) and BER from comm2.evaluate_hd_and_sd result dict.
    """
    q_db = np.nan
    ber = np.nan
    hd = res.get("HD", None)
    if hd is None:
        return q_db, ber
    try:
        cols = list(hd.columns)
        row = hd.loc["total"]
        if "Q_dB" in cols:
            q_db = float(row["Q_dB"])
        elif "Q" in cols:
            q_db = float(row["Q"])
        elif "QSq" in cols:
            q_db = float(row["QSq"])  # typical ~8-9dB in your plots
        if "BER" in cols:
            ber = float(row["BER"])
    except Exception:
        pass
    return q_db, ber


# =============================================================================
# Phase proxy: A_res = std(Δphi_blk) (optional but useful as x-axis)
# =============================================================================

@dataclass
class PhaseProxy:
    dphi_std_rad: float
    residual_rms_rad: float
    n_blocks: int

def compute_phase_proxy(
    y_eq: np.ndarray,
    x_ref: np.ndarray,
    block_len: int = 2048,
    eps: float = 1e-12,
) -> PhaseProxy:
    y = y_eq.reshape(-1)
    x = x_ref.reshape(-1)
    T = min(len(y), len(x))
    y = y[:T]
    x = x[:T]
    n_blk = T // block_len
    if n_blk < 2:
        return PhaseProxy(np.nan, np.nan, int(n_blk))

    phis = []
    for b in range(n_blk):
        ys = y[b * block_len:(b + 1) * block_len]
        xs = x[b * block_len:(b + 1) * block_len]
        zc = np.vdot(ys, xs)
        phis.append(np.angle(zc + eps))
    phis = np.unwrap(np.asarray(phis))

    dphi = np.diff(phis)
    dphi_std = float(np.std(dphi))

    t = np.arange(len(phis))
    a, b0 = np.polyfit(t, phis, 1)
    res = phis - (a * t + b0)
    res_rms = float(np.sqrt(np.mean(res ** 2)))

    return PhaseProxy(dphi_std, res_rms, int(n_blk))


# =============================================================================
# Forward once per checkpoint, then windowed evaluation
# =============================================================================

def forward_equalize_full(model: Any, params: FrozenDict, data: Any, L: int = 16):
    """
    One forward pass -> return (y_eq, x_ref) on full equalized span.
    """
    state, aux_inputs, const, sparams = model.initvar[1:]
    aux_inputs = core.dict_replace(aux_inputs, {"truth": data.x})

    params_net = util.dict_merge(params, sparams)
    z, _ = jax.jit(model.module.apply, backend="cpu")(
        {"params": params_net, "aux_inputs": aux_inputs, "const": const, **state},
        core.Signal(data.y),
    )

    x_ref = np.asarray(data.x[z.t.start:z.t.stop])
    y_eq = np.asarray(z.val)

    scale = comm2.qamscale(L) if L is not None else np.sqrt(10.0)
    return y_eq * scale, x_ref * scale


def eval_q_on_windows(
    y_eq: np.ndarray,
    x_ref: np.ndarray,
    windows: Sequence[Tuple[int, int]],
    L: int = 16,
    use_oracle_noise: bool = True,
    use_elliptical_llr: bool = True,
    temp_grid: Tuple[float, float, float] = (0.75, 1.0, 1.25),
    bitwidth: int = 6,
    decoder: Any = None,
    phase_block_len: int = 2048,
):
    if pd is None:
        raise ImportError("pandas is required for eval_q_on_windows()")

    records = []
    T = min(y_eq.shape[0], x_ref.shape[0])

    for wi, (s0, s1) in enumerate(windows):
        s0 = int(max(0, s0))
        s1 = int(min(T, s1))
        if s1 <= s0 + 256:
            continue

        y_w = y_eq[s0:s1]
        x_w = x_ref[s0:s1]

        y_1d = y_w.reshape(-1)
        x_1d = x_w.reshape(-1)

        sd_kwargs = dict(
            use_oracle_noise=use_oracle_noise,
            elliptical_llr=use_elliptical_llr,
            temp_grid=temp_grid,
            bitwidth=bitwidth,
            return_artifacts=False,
        )
        res = comm2.evaluate_hd_and_sd(
            y_1d, x_1d,
            L=L if L is not None else 16,
            decoder=decoder,
            sd_kwargs=sd_kwargs,
        )

        q_db, ber = _extract_q_ber(res)
        pp = compute_phase_proxy(y_w, x_w, block_len=phase_block_len)

        records.append(dict(
            win_id=int(wi), start=int(s0), stop=int(s1),
            Q_dB=float(q_db), BER=float(ber),
            A_res_dphi_std_rad=float(pp.dphi_std_rad),
            phi_blk_residual_rms_rad=float(pp.residual_rms_rad),
            cpe_n_blocks=int(pp.n_blocks),
        ))

    return pd.DataFrame.from_records(records)


def summarize_windows(df_win, q_quantile: float = 0.10):
    if df_win is None or len(df_win) == 0:
        return dict(
            Q_min=np.nan, Q_q10=np.nan, Q_median=np.nan, Q_mean=np.nan, Q_max=np.nan,
            A_res_median=np.nan, phi_res_median=np.nan,
        )
    q = df_win["Q_dB"].to_numpy()
    return dict(
        Q_min=float(np.nanmin(q)),
        Q_q10=float(np.nanquantile(q, q_quantile)),
        Q_median=float(np.nanmedian(q)),
        Q_mean=float(np.nanmean(q)),
        Q_max=float(np.nanmax(q)),
        A_res_median=float(np.nanmedian(df_win["A_res_dphi_std_rad"].to_numpy())),
        phi_res_median=float(np.nanmedian(df_win["phi_blk_residual_rms_rad"].to_numpy())),
    )


def boundary_over_checkpoints(
    model: Any,
    checkpoints: Sequence[Tuple[int, FrozenDict]],
    data_eval: Any,
    windows: Sequence[Tuple[int, int]],
    L: int = 16,
    use_oracle_noise: bool = True,
    use_elliptical_llr: bool = True,
    temp_grid: Tuple[float, float, float] = (0.75, 1.0, 1.25),
    bitwidth: int = 6,
    decoder: Any = None,
    phase_block_len: int = 2048,
    q_quantile: float = 0.10,
):
    if pd is None:
        raise ImportError("pandas is required for boundary_over_checkpoints()")

    sum_records = []
    for it, params in checkpoints:
        y_eq, x_ref = forward_equalize_full(model, params, data_eval, L=L)
        df_win = eval_q_on_windows(
            y_eq, x_ref, windows=windows, L=L,
            use_oracle_noise=use_oracle_noise,
            use_elliptical_llr=use_elliptical_llr,
            temp_grid=temp_grid, bitwidth=bitwidth, decoder=decoder,
            phase_block_len=phase_block_len,
        )
        stats = summarize_windows(df_win, q_quantile=q_quantile)
        stats["iter"] = int(it)
        sum_records.append(stats)

    return pd.DataFrame.from_records(sum_records).sort_values("iter").reset_index(drop=True)


def plot_fig2_q_boundary(
    df_base,
    df_ours,
    q_fail_thr: Optional[float] = None,
    title: str = "Figure 2: Where it fails (Q-boundary, fixed switch=200k)",
    show: bool = True,
) -> plt.Figure:
    if pd is None:
        raise ImportError("pandas is required for plot_fig2_q_boundary()")

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for df, name in [(df_base, "baseline"), (df_ours, "ours")]:
        ax1.plot(df["iter"], df["Q_q10"], marker="x", linestyle="-", label=f"{name} Q_0.1")
        ax1.plot(df["iter"], df["Q_min"], marker="o", linestyle="--", label=f"{name} Q_min")

    if q_fail_thr is not None:
        ax1.axhline(float(q_fail_thr), linestyle=":", label=f"fail thr = {q_fail_thr:.2f} dB")

    ax1.set_xlabel("iteration")
    ax1.set_ylabel("Q (dB)")
    ax1.set_title("Robust / worst-case Q over time windows")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for df, name, mk in [(df_base, "baseline", "o"), (df_ours, "ours", "x")]:
        ax2.plot(df["A_res_median"], df["Q_q10"], marker=mk, linestyle="-", label=f"{name} Q_0.1")

    if q_fail_thr is not None:
        ax2.axhline(float(q_fail_thr), linestyle=":")

    ax2.set_xlabel(r"$A_{res}$ proxy (median std($\Delta\phi_{blk}$)) [rad]")
    ax2.set_ylabel("Q_0.1 (dB)")
    ax2.set_title("Boundary in (proxy, Q) plane")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
