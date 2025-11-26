#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
from typing import Optional, Tuple

# =========================================
# 固定設定（DTLZ2, m=2, 1回実行）
# =========================================
prob      = "DTLZ2"
m         = 2
t         = 1                  # 参照点ファイルの type
r_radius  = 0.10               # 実スケールでの楕円半径ベクトル r⃗ = nadir * r_radius
dpi_save  = 600
out_dir   = f'../output/roi_from_pf/{prob}/m{m}/'

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

# =========================================
# Utility
# =========================================
def load_pf(prob: str, m: int, t: int) -> np.ndarray:
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob: str, m: int, t: int) -> np.ndarray:
    z_path = f'../ref_point_data/m{m}_{prob}_type{t}.csv'
    return np.loadtxt(z_path, delimiter=',', ndmin=1)

def make_endpoints_formatter(ideal_val: float, nadir_val: float):
    def _fmt(x, pos=None):
        if np.isclose(x, 0.0): return rf"${ideal_val:g}$"
        if np.isclose(x, 1.0): return rf"${nadir_val:g}$"
        return ""
    return _fmt

def normalize_points(X: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    denom = np.where(np.abs(nadir - ideal) < 1e-12, 1.0, nadir - ideal)
    return (X - ideal) / denom

# =========================================
# ROI 抽出（PF → S′）
# =========================================
def _closest_pf_index(PF: np.ndarray, z: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(PF - z, axis=1)))

def _aug_chebyshev_asf(p_n: np.ndarray, z_n: np.ndarray, w: np.ndarray, rho: float = 1e-6) -> float:
    d = np.abs(p_n - z_n)
    return float(np.max(w * d) + rho * np.sum(w * d))

def _argmin_asf_idx(PF: np.ndarray, z: np.ndarray, ideal: np.ndarray, nadir: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> int:
    PF_n = normalize_points(PF, ideal, nadir)
    z_n  = normalize_points(z,  ideal, nadir)
    if weights is None:
        weights = np.ones(PF.shape[1], dtype=float)
    w = weights / np.sum(weights)
    vals = np.array([_aug_chebyshev_asf(p, z_n, w) for p in PF_n])
    return int(np.argmin(vals))

def _dominates_min(p: np.ndarray, q: np.ndarray) -> bool:
    # 最小化の Pareto 支配: p ≺ q
    return np.all(p <= q) and np.any(p < q)

def extract_roi_from_pf(
    PF: np.ndarray,
    z: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    r_vec: np.ndarray | float,
    mode: str = "C",                  # "C" | "A" | "P"
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    PF から ROI（S'）を抽出。
    Returns:
        S_prime : (k, m)  抽出サブセット
        mask    : (|PF|,) bool  ROI 内マスク
        center  : (m,) or None  ROI-C/A の中心（描画用）
    """
    m = PF.shape[1]
    if np.isscalar(r_vec): r_vec = np.full(m, float(r_vec))
    r_vec = np.asarray(r_vec, dtype=float)

    mode = mode.upper()
    if mode == "C":
        idx = _closest_pf_index(PF, z)
        center = PF[idx]
        diff = PF - center
        mask = np.sum((diff / r_vec) ** 2, axis=1) <= 1.0
        return PF[mask], mask, center

    if mode == "A":
        idx = _argmin_asf_idx(PF, z, ideal, nadir, weights)
        center = PF[idx]
        diff = PF - center
        mask = np.sum((diff / r_vec) ** 2, axis=1) <= 1.0
        return PF[mask], mask, center

    if mode == "P":
        PF_n = normalize_points(PF, ideal, nadir)
        z_n  = normalize_points(z,  ideal, nadir)
        dom_pf = np.array([_dominates_min(p, z_n) for p in PF_n])  # p ≺ z
        mask = dom_pf if np.any(dom_pf) else np.array([_dominates_min(z_n, p) for p in PF_n])  # z ≺ p
        return PF[mask], mask, None

    raise ValueError("mode must be one of {'C','A','P'}")

# =========================================
# 描画（PFのみ／ROIハイライト）
# =========================================
def _setup_axes(ax, ideal: np.ndarray, nadir: np.ndarray):
    ax.set_xlabel(r'$f_1$', fontsize=50)
    ax.set_ylabel(r'$f_2$', fontsize=50)
    ax.set_xlim([0, 1 + 0.1])
    ax.set_ylim([0, 1 + 0.1])
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.xaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(ideal[0], nadir[0])))
    ax.yaxis.set_major_formatter(FuncFormatter(make_endpoints_formatter(ideal[1], nadir[1])))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=45, length=0)
    ax.tick_params(axis='both', which='minor', labelsize=45, length=0)
    for spine in ax.spines.values(): spine.set_linewidth(2.5)
    plt.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.25)

def draw_roi_only(PF: np.ndarray, z: np.ndarray, mode: str, weights: Optional[np.ndarray] = None):
    ideal, nadir = PF.min(axis=0), PF.max(axis=0)
    PF_n = normalize_points(PF, ideal, nadir)
    z_n  = normalize_points(z,  ideal, nadir)

    # 元スケール楕円半径ベクトル r⃗
    r_vec = nadir * r_radius

    # 抽出
    S_prime, mask, center = extract_roi_from_pf(PF, z, ideal, nadir, r_vec=r_vec, mode=mode, weights=weights)
    S_n = normalize_points(S_prime, ideal, nadir) if S_prime.size else np.empty((0, 2))

    # 図
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _setup_axes(ax, ideal, nadir)

    # PF（薄）
    ax.scatter(PF_n[:, 0], PF_n[:, 1], s=3, alpha=0.2, color='black', rasterized=True)

    # ROI 部分（濃）
    if S_n.size > 0:
        ax.scatter(S_n[:, 0], S_n[:, 1], s=100,rasterized=True)

    # 参照点 z
    ax.scatter(z_n[0], z_n[1], s=230, marker='^', color=(44/255,160/255,44/255), zorder=4)

    # ROI-C/A: 中心＋楕円（正規化後の半径で正確に描画）
    if center is not None:
        c_n = normalize_points(center, ideal, nadir)
        ax.scatter(c_n[0], c_n[1], s=100, marker='s', color=(255/255,127/255,14/255), zorder=4)

        # 正規化半径ベクトル r⃗_norm = r⃗ / (nadir - ideal)
        denom = np.where(np.abs(nadir - ideal) < 1e-12, 1.0, nadir - ideal)
        r_norm = (r_vec / denom)
        ax.add_patch(
            Ellipse(
                xy=(c_n[0], c_n[1]),
                width=2 * r_norm[0],
                height=2 * r_norm[1],
                fill=False, edgecolor='black',
                linestyle=(0, (1.9, 1)), linewidth=1.5
            )
        )

    # ROI-P: 補助線（z の垂直・水平）
    if mode.upper() == "P":
        # 現在の軸範囲を取得
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # x = z_n[0] の縦線（y が大きい側のみ）
        ax.plot([z_n[0], z_n[0]], [z_n[1], ylim[1]], linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')

        # y = z_n[1] の横線（x が大きい側のみ）
        ax.plot([z_n[0], xlim[1]], [z_n[1], z_n[1]], linestyle=(0, (1.9, 1)), linewidth=1.5, color='black')

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    tag = {"C": "roi_c", "A": "roi_a", "P": "roi_p"}[mode.upper()]
    plt.savefig(os.path.join(out_dir, f'final_{tag}.pdf'), dpi=dpi_save)
    plt.savefig(os.path.join(out_dir, f'final_{tag}.png'), dpi=dpi_save)
    plt.close(fig)
    print(f"[{mode}] saved -> {out_dir}final_{tag}.[pdf|png]")

# =========================================
# 実行（PFとzのみ）
# =========================================
def main():
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    z_path  = f'../ref_point_data/m{m}_{prob}_type{t}.csv'

    # if not (os.path.exists(pf_path) and os.path.exists(z_path)):
    #     print("❌ PF か z の入力ファイルが見つかりません。")
    #     print(f"  PF: {pf_path}")
    #     print(f"  z : {z_path}")
    #     return

    PF = np.loadtxt(pf_path, delimiter=',')
    # z  = np.loadtxt(z_path, delimiter=',', ndmin=1)
    z = np.array([0.5,0.5])
    # ROI-C / ROI-A / ROI-P を順に描画
    for mode in ["C", "A", "P"]:
        draw_roi_only(PF, z, mode=mode)

if __name__ == "__main__":
    main()
