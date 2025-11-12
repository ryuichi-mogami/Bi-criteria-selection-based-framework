#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

# =========================
# ユーザー設定（3目的専用）
# =========================
n_obj_list   = [3]
problems     = ["DTLZ2"]
roi_types    = ["roi-c"]     # 使用: 'emo', 'roi-c', 'roi-p'
algorithms   = ['BNSGA2']
pop_sel_caption = {'POP': 'TRUE'}
pop_selection = list(pop_sel_caption.keys())

n_runs     = 31
t          = 1
r_radius   = 0.1
mu         = 100
out_dir    = '../output/3d_image_median'
dpi_save   = 600
# 正規化空間での可視化半径（ROI-Cの円用）
roi_r_norm = 0.08

# ===== 2D投影ビュー設定 =====
VIEW_USE_ANGLES = True      # 角度指定を使う
VIEW_ELEV_DEG   = 20.0      # 仰角（度）
VIEW_AZIM_DEG   = -15.0      # 方位（度）
UP_HINT         = np.array([0.0, 0.0, 1.0])   # 画面の「上」方向のヒント

# ---------- Utility ----------
def normalize_points(X, ideal, nadir):
    denom = (nadir - ideal)
    return (X - ideal) / denom

def normal_from_angles(elev_deg, azim_deg):
    """ELEV/AZIM から視線方向（=平面法線）ベクトルを作る"""
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    n = np.array([np.cos(elev)*np.cos(azim),
                  np.cos(elev)*np.sin(azim),
                  np.sin(elev)])
    return n / (np.linalg.norm(n) + 1e-12)

# ---------- IGD 系（使用分のみ） ----------
def compute_igd_c_plus(X, PF, PF_norm, z, r, ideal, nadir, prob, m):
    distance_list = np.zeros(len(PF))
    for i, p in enumerate(PF):
        distance_list[i] = np.linalg.norm(p - z)

    pivot_dir = "../output/pivot"
    os.makedirs(pivot_dir, exist_ok=True)
    pivot_file = os.path.join(pivot_dir, f"{prob}_{m}.csv")

    if os.path.exists(pivot_file):
        pivot_id = int(np.loadtxt(pivot_file))
    else:
        pivot_id = int(np.argmin(distance_list))
        np.savetxt(pivot_file, [pivot_id], fmt='%d')
    pivot_point = PF[pivot_id]

    diff = PF - pivot_point
    val = np.sum((diff / r) ** 2, axis=1)   # 超球（3D）
    mask = val <= 1.0
    S_prime = PF[mask]

    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))

def compute_igd_p_plus(X, PF, PF_norm, z, r, ideal, nadir):
    less_eq = np.all(PF <= z, axis=1)
    greater_eq = np.all(PF >= z, axis=1)
    mask = np.logical_or(less_eq, greater_eq)
    S_prime = PF[mask]

    if S_prime.shape[0] == 0:
        return np.nan
    igd_vals = []
    for s in S_prime:
        diff = X - s
        diff_pos = np.maximum(diff, 0)
        dists = np.linalg.norm(diff_pos, axis=1)
        igd_vals.append(dists.min())
    return float(np.mean(igd_vals))

# ---------- 入出力 ----------
def sol_path(pop_sel, roi_type, alg, prob, m, run):
    base = f'../output/results/{roi_type}/{alg}/{prob}/m{m}/'
    base_igd = f'../output/igdC_plus/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(base, exist_ok=True)
    os.makedirs(base_igd, exist_ok=True)
    csv_path = f'{base}pop_{run}th_run_50000fevals.csv'
    igd_path = f'{base_igd}pop_{run}th_run_50000fevals.csv'
    return csv_path, igd_path

def load_pf(prob, m, t):
    pf_path = f'../ref_point_dataset/{prob}_d{m}_n50000.csv'
    return np.loadtxt(pf_path, delimiter=',')

def load_ref_point(prob, m, t):
    return np.loadtxt(f'../ref_point_data/roi-c/m{m}_{prob}_type{t}.csv',
                      delimiter=',', ndmin=1)

def select_median_run(vals):
    valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if not valid:
        return None
    sorted_vals = sorted(valid, key=lambda x: x[1])
    n = len(sorted_vals)
    target = sorted_vals[n//2][1] if n % 2 else 0.5*(sorted_vals[n//2-1][1]+sorted_vals[n//2][1])
    best = min(sorted_vals, key=lambda x: (abs(x[1]-target), x[0]))
    return best[0]

# ---------- ROI マスク ----------
def mask_pf_in_true_roi(PF_norm, roi_type, nearest_point, ref_norm, roi_r_norm):
    if roi_type == 'roi-c':
        d = np.linalg.norm(PF_norm - nearest_point, axis=1)
        return (d <= roi_r_norm)
    elif roi_type == 'roi-p':
        less_eq    = np.all(PF_norm <= ref_norm, axis=1)
        greater_eq = np.all(PF_norm >= ref_norm, axis=1)
        return np.logical_or(less_eq, greater_eq)
    else:  # emo
        return np.zeros(len(PF_norm), dtype=bool)

# ---------- 直交投影ユーティリティ ----------
def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

def make_projection_basis(n, up_hint=np.array([0.0, 0.0, 1.0])):
    """
    法線 n を持つ平面上の直交基底 (u, v, n) を返す。
    v は up_hint の平面射影方向に揃える（= 画像の「上」）。
    """
    n = _normalize(np.asarray(n, float))
    up = np.asarray(up_hint, float)
    up_proj = up - (up @ n) * n
    if np.linalg.norm(up_proj) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        up_proj = up - (up @ n) * n
    v = _normalize(up_proj)
    u = _normalize(np.cross(n, v))
    if np.dot(np.cross(u, v), n) < 0:
        v = -v
    return u, v, n

def project_points_to_plane_with_basis(P, p0, basis):
    """P(?,3) を平面 (p0, basis=(u,v,n)) に直交投影 → 2D (u,v) を返す"""
    u, v, n = basis
    W = P - p0
    U = W @ u
    V = W @ v
    return np.stack([U, V], axis=1)

def project_segment_to_plane_with_basis(p_start, p_end, p0, basis):
    pts = np.vstack([p_start, p_end])
    uv = project_points_to_plane_with_basis(pts, p0, basis)
    return uv[0], uv[1]

def cube_uv_bounds_with_basis(p0, basis, pad=0.05):
    """[0,1]^3 の8頂点を投影してプロット枠を決める"""
    corners = np.array([[x, y, z] for x in [0,1] for y in [0,1] for z in [0,1]])
    uv = project_points_to_plane_with_basis(corners, p0, basis)
    (umin, vmin), (umax, vmax) = uv.min(0), uv.max(0)
    du, dv = umax-umin, vmax-vmin
    return (umin - pad*du, umax + pad*du, vmin - pad*dv, vmax + pad*dv)

# ---- 反転を戻す関数（削除しない）----
def enforce_axis_orientation(basis, p0):
    """
    2D上で +f1 が右(+u), +f3 が上(+v) になるように u, v の符号を調整。
    右手系 (det>0) を維持/回復する。
    """
    u, v, n = basis

    zero_uv = project_points_to_plane_with_basis(np.zeros((1,3)), p0, basis)[0]
    e1_uv   = project_points_to_plane_with_basis(np.array([[1.0,0.0,0.0]]), p0, basis)[0] - zero_uv
    e3_uv   = project_points_to_plane_with_basis(np.array([[0.0,0.0,1.0]]), p0, basis)[0] - zero_uv

    # +f1 が右（u正）に行くように
    if e1_uv[0] < 0:
        u = -u

    # +f3 が上（v正）に行くように
    if e3_uv[1] < 0:
        v = -v

    # 右手系を担保（u×v が n と同向）
    if np.dot(np.cross(u, v), n) < 0:
        u = -u

    return u, v, n

# ---------- ROI 可視化（投影） ----------
def draw_roi_p_cross_lines_projected(ax2, ref_norm, p0, basis, lw=1.5, zorder=3):
    segs = [
        (np.array([0.0, ref_norm[1], ref_norm[2]]), np.array([ref_norm[0], ref_norm[1], ref_norm[2]])),
        (np.array([ref_norm[0], 0.0, ref_norm[2]]), np.array([ref_norm[0], ref_norm[1], ref_norm[2]])),
        (np.array([ref_norm[0], ref_norm[1], 0.0]), np.array([ref_norm[0], ref_norm[1], ref_norm[2]])),
    ]
    for a3, b3 in segs:
        a2, b2 = project_segment_to_plane_with_basis(a3, b3, p0, basis)
        ax2.plot([a2[0], b2[0]], [a2[1], b2[1]],
                 linestyle=(0, (1.9, 1)), linewidth=lw, color='black', zorder=zorder)

def draw_roi_c_circle_projected(ax2, center3, radius, p0, basis, lw=1.0, zorder=3):
    center2 = project_points_to_plane_with_basis(center3[None, :], p0, basis)[0]
    theta = np.linspace(0, 2*np.pi, 400)
    circle = np.c_[np.cos(theta), np.sin(theta)] * radius + center2
    ax2.plot(circle[:, 0], circle[:, 1],
             color='black', linewidth=lw, alpha=0.8,
             linestyle=(0, (1.9, 1)), zorder=zorder)

def draw_unit_cube_back_faces_projected(ax2, p0, basis, n_vec,
                                        ticks=(0.0, 0.5, 1.0),
                                        edge_lw=1.6, edge_alpha=0.95,
                                        grid_lw=0.8, grid_alpha=0.35):
    """
    [0,1]^3 のうち、視点方向 n_vec に対して背面となる3面だけ投影して描画。
    """
    inner = [t for t in ticks if 0.0 < t < 1.0]

    def _seg(a3, b3, lw, alpha, z):
        a2, b2 = project_segment_to_plane_with_basis(np.array(a3), np.array(b3), p0, basis)
        ax2.plot([a2[0], b2[0]], [a2[1], b2[1]],
            color='black', linewidth=lw, alpha=alpha, zorder=z)

    faces = [
        ('x', 0.0, np.array([-1, 0, 0])), ('x', 1.0, np.array([+1, 0, 0])),
        ('y', 0.0, np.array([0, -1, 0])), ('y', 1.0, np.array([0, +1, 0])),
        ('z', 0.0, np.array([0, 0, -1])), ('z', 1.0, np.array([0, 0, +1])),
    ]

    back_faces = [(axname, val) for (axname, val, n_face) in faces
                  if float(np.dot(n_face, n_vec)) < 0.0]
    for axname, val in back_faces:
        if axname == 'x':
            x = val
            _seg([x,0,0],[x,1,0], edge_lw, edge_alpha, 2)
            _seg([x,0,1],[x,1,1], edge_lw, edge_alpha, 2)
            _seg([x,0,0],[x,0,1], edge_lw, edge_alpha, 2)
            _seg([x,1,0],[x,1,1], edge_lw, edge_alpha, 2)
            for t in inner:
                _seg([x,t,0],[x,t,1], grid_lw, grid_alpha, 1)
                _seg([x,0,t],[x,1,t], grid_lw, grid_alpha, 1)
        elif axname == 'y':
            y = val
            _seg([0,y,0],[1,y,0], edge_lw, edge_alpha, 2)
            _seg([0,y,1],[1,y,1], edge_lw, edge_alpha, 2)
            _seg([0,y,0],[0,y,1], edge_lw, edge_alpha, 2)
            _seg([1,y,0],[1,y,1], edge_lw, edge_alpha, 2)
            for t in inner:
                _seg([t,y,0],[t,y,1], grid_lw, grid_alpha, 1)
                _seg([0,y,t],[1,y,t], grid_lw, grid_alpha, 1)
        else:  # 'z'
            z = val
            _seg([0,0,z],[1,0,z], edge_lw, edge_alpha, 2)
            _seg([0,1,z],[1,1,z], edge_lw, edge_alpha, 2)
            _seg([0,0,z],[0,1,z], edge_lw, edge_alpha, 2)
            _seg([1,0,z],[1,1,z], edge_lw, edge_alpha, 2)
            for t in inner:
                _seg([t,0,z],[t,1,z], grid_lw, grid_alpha, 1)
                _seg([0,t,z],[1,t,z], grid_lw, grid_alpha, 1)

def label_axes_on_front_edges(ax2, p0, basis,
                              fs=18, center_offset=0.05, radial_offset=0.1):
    """
    各軸の中間点から少し外側に f1, f2, f3 ラベルを配置。
    f1: (1,1,0)->(0,1,0)
    f2: (1,0,0)->(1,1,0)
    f3: (1,0,0)->(1,0,1)
    """
    proj = lambda pt: project_points_to_plane_with_basis(np.asarray(pt)[None, :], p0, basis)[0]
    center = proj([0.5, 0.5, 0.5])

    def place_label(a3, b3, text):
        a = proj(a3); b = proj(b3)
        mid = (a + b) / 2.0
        d = b - a
        L = np.linalg.norm(d) + 1e-12
        t = d / L
        # 軸方向に少し外へ
        P = mid + t * (center_offset * L)
        # さらに外側へ逃がす
        out = (P - center); out /= (np.linalg.norm(out) + 1e-12)
        P += out * (radial_offset * L)
        ax2.text(P[0], P[1], text, fontsize=fs, ha="center", va="center", color="#222", zorder=6)

    place_label([1,0,0], [0,0,0], r"$f_1$")
    place_label([1,0,0], [1,1,0], r"$f_2$")
    place_label([0,0,0], [0,0,1], r"$f_3$")


def draw_axis_fixed_labels(ax2,
                           fpos=None,            # 軸名の位置
                           tpos=None,            # 端値の位置
                           fs=30, halo=True,clip=False):
    """
    画面固定位置(axes座標系)に f1,f2,f3 の軸名と 0/1.0 を描く。
    - fpos: {'f1':(x,y), 'f2':(x,y), 'f3':(x,y)}  in ax2.transAxes
    - tpos: {'f1':{'0':(x,y), '1':(x,y)}, ...}    in ax2.transAxes
    """

    # デフォルト（例：f3を左側に、f1は手前左、f2は手前中央）
    if fpos is None:
        fpos = {
            'f1': (0.015, 0.13),
            'f2': (0.6, -0.03),
            'f3': (-0.1, 0.57)   # 左側
        }
    if tpos is None:
        tpos = {
            'f1': {'0': (0.04, 0.23), '0.5': (0.11, 0.13),'1': (0.19, 0.03)},   # 画像の「0 1.0」を逆にしたいならここを入れ替える
            'f2': {'0': (0.3, 0.03), '0.5': (0.6, 0.04),'1': (0.98, 0.07)},
            'f3': {'0': (0.01, 0.3), '0.5': (0, 0.57),'1': (0, 0.85)},   # 左側縦に 0→下, 1→上
        }

    peff = [pe.withStroke(linewidth=3, foreground="white", alpha=0.9)] if halo else None

    # 軸名
    for k, (x, y) in fpos.items():
        ax2.text(x, y, fr"$f_{k[-1]}$", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=fs, zorder=6,
                 path_effects=peff,clip_on=clip)

    # 端値
    for k, d in tpos.items():
        x0, y0 = d['0']
        x1, y1 = d['0.5']
        x2, y2 = d['1']
        ax2.text(x0, y0, "0",   transform=ax2.transAxes,
                 ha='center', va='center', fontsize=fs-1, color="#444",
                 zorder=6, path_effects=peff,clip_on=clip)
        ax2.text(x1, y1, "0.5", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=fs-1, color="#444",
                 zorder=6, path_effects=peff,clip_on=clip)
        ax2.text(x2, y2, "1.0", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=fs-1, color="#444",
                 zorder=6, path_effects=peff,clip_on=clip)

def draw_axis_end_values_on_edges(ax2, ideal, nadir, p0, basis,
                                  fs=16, along_pts=8, out_pts=10,
                                  halo=True, avoid_overlap=True):
    """
    端(ideal/nadir)ラベルを、画面座標(pt)で軸方向/外向きにオフセットして配置。
    多少の重なりは自動で外側へ押し出します。
      - along_pts: 軸方向のずらし量 [pt]
      - out_pts  : 外向きのずらし量 [pt]
    """
    proj = lambda pt: project_points_to_plane_with_basis(np.asarray(pt)[None, :], p0, basis)[0]
    center = proj([0.5, 0.5, 0.5])

    fig = ax2.figure
    renderer = fig.canvas.get_renderer()
    placed_boxes = []

    def _add_text(xy, s, dx_pt, dy_pt, ha, va):
        # 画面座標(pt)での平行移動
        trans = ax2.transData + mtransforms.ScaledTranslation(dx_pt/72, dy_pt/72, fig.dpi_scale_trans)
        txt = ax2.text(xy[0], xy[1], s, fontsize=fs, ha=ha, va=va,
                       transform=trans, clip_on=False, zorder=6,
                       path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.9)] if halo else None)

        if avoid_overlap:
            # 一度描画して bbox を取得
            fig.canvas.draw()
            bb = txt.get_window_extent(renderer=renderer).expanded(1.05, 1.15)

            # 既存と重なるなら外向き方向に段階的に押し出す
            bump = 0
            while any(bb.overlaps(p) for p in placed_boxes) and bump < 5:
                bump += 1
                trans2 = ax2.transData + mtransforms.ScaledTranslation(
                    dx_pt/72, (dy_pt + bump*1.2*out_pts)/72, fig.dpi_scale_trans
                )
                txt.set_transform(trans2)
                fig.canvas.draw()
                bb = txt.get_window_extent(renderer=renderer).expanded(1.05, 1.15)

            placed_boxes.append(bb)

    def place_pair(a3, b3, v0, v1):
        a = proj(a3); b = proj(b3)
        mid = (a + b) / 2.0

        # 軸方向ベクトル（2D）
        t = b - a
        Lt = np.linalg.norm(t) + 1e-12
        t = t / Lt

        # 「外向き」= 中心→辺の中点 方向
        out = mid - center
        Lo = np.linalg.norm(out) + 1e-12
        out = out / Lo

        # 文字揃えは方向で自動
        ha_a = 'right' if t[0] >= 0 else 'left'
        ha_b = 'left'  if t[0] >= 0 else 'right'
        # 上下は外向きの y で決定（図の上側/下側へずらす）
        va_common = 'bottom' if out[1] >= 0 else 'top'

        # 端点 a=ideal 側 / b=nadir 側
        _add_text(a, f"{v0:g}",
                  dx_pt=(-along_pts if t[0] >= 0 else along_pts),
                  dy_pt=( out_pts if out[1] >= 0 else -out_pts),
                  ha=ha_a, va=va_common)

        _add_text(b, f"{v1:.1f}",
                  dx_pt=( along_pts if t[0] >= 0 else -along_pts),
                  dy_pt=( out_pts if out[1] >= 0 else -out_pts),
                  ha=ha_b, va=va_common)

    # f1: (0,0,0)->(1,0,0) / f2: (1,0,0)->(1,1,0) / f3: (0,0,0)->(0,0,1)
    place_pair([0,0,0], [1,0,0], ideal[0], nadir[0])   # f1
    place_pair([1,0,0], [1,1,0], ideal[1], nadir[1])   # f2
    place_pair([0,0,0], [0,0,1], ideal[2], nadir[2])   # f3

# ---------- 2D（平面直交投影）描画 ----------
USE_2D_PROJECTION = True
PROJ_PLANE_ORIGIN = "ref"  # "ref" or "pivot"

def plot_2d_projected(prob, m, roi_type, alg, pop_sel, run, PF, Pset, z, n_obj):
    # 正規化
    I, N = PF.min(axis=0), PF.max(axis=0)
    PF_norm  = normalize_points(PF,  I, N)
    P_norm   = normalize_points(Pset, I, N)
    ref_norm = normalize_points(z,    I, N)
    ideal = np.min(PF, axis=0)
    nadir = np.max(PF, axis=0)
    # ピボット（ROI-C/ROI-P の中心決め）
    if roi_type in ('roi-c', 'roi-p'):
        pivot_id = int(np.argmin(np.linalg.norm(PF - z, axis=1)))
        nearest_point = PF_norm[pivot_id]
    else:
        nearest_point = None

    # 投影平面 (p0, n_vec) と基底（+f3 を上に固定）
    p0 = ref_norm if PROJ_PLANE_ORIGIN == "ref" else (nearest_point if nearest_point is not None else np.array([1/3,1/3,1/3]))
    n_vec = normal_from_angles(VIEW_ELEV_DEG, VIEW_AZIM_DEG) if VIEW_USE_ANGLES else np.array([1.0,1.0,1.0])
    basis = make_projection_basis(n_vec, up_hint=UP_HINT)
    # ← 反転を戻す関数を必ず適用
    basis = enforce_axis_orientation(basis, p0)

    # 直交投影
    PF_uv  = project_points_to_plane_with_basis(PF_norm[:, :3], p0, basis)
    P_uv   = project_points_to_plane_with_basis(P_norm[:, :3],   p0, basis)
    ref_uv = project_points_to_plane_with_basis(ref_norm[None, :3], p0, basis)[0]

    # 図
    fig, ax2 = plt.subplots(figsize=(6.8, 6.8))

    # 軸範囲（立方体の8頂点で安定化）
    umin, umax, vmin, vmax = cube_uv_bounds_with_basis(p0, basis, pad=0.07)
    ax2.set_xlim(umin, umax); ax2.set_ylim(vmin, vmax)
    ax2.set_aspect('equal', adjustable='box')

    # 背景（目盛・スパイン非表示）
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values(): spine.set_visible(False)

    # ---- レイヤ順に描画 ----
    # 1) PF（背景）
    ax2.scatter(PF_uv[:, 0], PF_uv[:, 1],
                s=10, color=(0.72,0.72,0.72), alpha=0.09, edgecolors='none',
                zorder=1)

    # 2) 真の ROI 内 PF を強調
    if roi_type != 'emo':
        roi_mask = mask_pf_in_true_roi(
            normalize_points(PF, I, N), roi_type,
            nearest_point if nearest_point is not None else ref_norm,
            ref_norm, roi_r_norm
        )
        if np.any(roi_mask):
            uv_roi = PF_uv[roi_mask]
            ax2.scatter(uv_roi[:, 0], uv_roi[:, 1],
                        s=16, color="gray", alpha=0.28, edgecolors='none',
                        zorder=2)

    # 3) ROI 可視化（破線）
    # if roi_type == 'roi-c' and nearest_point is not None:
    #     draw_roi_c_circle_projected(ax2, nearest_point, roi_r_norm, p0, basis, lw=1.0, zorder=4)
    # el
    if roi_type == 'roi-p':
        draw_roi_p_cross_lines_projected(ax2, ref_norm, p0, basis, lw=1.5, zorder=4)

    # 4) 解集合（青を最前面）
    ax2.scatter(P_uv[:, 0], P_uv[:, 1],
                s=80, color=(31/255,119/255,180/255), edgecolors='none',
                zorder=3)

    # 5) 参照点（三角）
    if roi_type != 'emo':
        ax2.scatter([ref_uv[0]], [ref_uv[1]],
                    s=150, marker='^', color=(44/255,160/255,44/255),
                    zorder=4)

    # 6) 背面の3面枠＋前面ラベル＋端値
    draw_unit_cube_back_faces_projected(ax2, p0, basis, n_vec,
                                        ticks=(0.0, 0.5, 1.0),
                                        edge_lw=1.6, grid_lw=0.8)
    # label_axes_on_front_edges(ax2, p0, basis, fs=25)
    # draw_axis_end_values_on_edges(ax2, ideal, nadir, p0, basis, fs=18)
    draw_axis_fixed_labels(ax2)
    # 保存
    output_dir = f'{out_dir}/{roi_type}/{alg}/{prob}/m{m}/'
    os.makedirs(output_dir, exist_ok=True)
    image_file_pdf = os.path.join(output_dir, 'final_pop_median_run_projected.pdf')
    image_file_png = os.path.join(output_dir, 'final_pop_median_run_projected.png')

    plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.07)
    plt.savefig(image_file_pdf, dpi=dpi_save)
    plt.savefig(image_file_png, dpi=dpi_save)
    plt.close(fig)
    print(f"2D投影プロット画像を作成しました: {image_file_pdf}")
    print(f"2D投影プロット画像を作成しました: {image_file_png}")

# 追加（基準にするpop_sel）
REF_POP = 'POP'
n = [50000, 50086, 102340, 455126, 3162510]  # m=3 → n[1]=50086 を使用

# ---------- メイン ----------
def main():
    os.makedirs(out_dir, exist_ok=True)
    for m in n_obj_list:
        for prob in problems:
            pf_path = f'../ref_point_dataset/{prob}_d{m}_n{n[m - 2]}.csv'
            pf_npy = pf_path.replace('.csv', '.npy')
            if not os.path.exists(pf_npy):
                PF = load_pf(prob, m, t)
                np.save(pf_npy, PF)
            else:
                PF = np.load(pf_npy)
            z  = load_ref_point(prob, m, t)
            ideal, nadir = PF.min(axis=0), PF.max(axis=0)
            PF_norm = normalize_points(PF, ideal, nadir)

            for roi_type in roi_types:
                r_radius_elipse = np.array([nadir[i] * r_radius for i in range(m)])

                for alg in algorithms:
                    # 1) まず基準POPで中央値runを決める
                    ref_vals = []
                    for run in range(n_runs):
                        ref_csv, ref_igd = sol_path(REF_POP, roi_type, alg, prob, m, run)
                        if not os.path.exists(ref_csv):
                            continue
                        if os.path.exists(ref_igd):
                            igd = float(np.loadtxt(ref_igd))
                        else:
                            X = np.loadtxt(ref_csv, delimiter=',', ndmin=2)
                            if roi_type == 'roi-c':
                                igd = compute_igd_c_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, prob, m)
                            elif roi_type == 'roi-p':
                                igd = compute_igd_p_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir)
                            elif roi_type == 'emo':
                                igd = compute_igd_c_plus(X, PF, PF_norm, z, r_radius_elipse, ideal, nadir, prob, m)
                            np.savetxt(ref_igd, [igd])
                        ref_vals.append((run, igd))

                    if not ref_vals:
                        continue

                    ref_median_run = select_median_run([v for _, v in ref_vals])
                    if ref_median_run is None:
                        continue

                    # 2) 決めたrun番号で描画
                    for pop_sel in pop_selection:
                        sol_csv, _ = sol_path(pop_sel, roi_type, alg, prob, m, ref_median_run)
                        if not os.path.exists(sol_csv):
                            continue
                        Pset = np.loadtxt(sol_csv, delimiter=',', ndmin=2)
                        plot_2d_projected(prob, m, roi_type, alg, pop_sel, ref_median_run, PF, Pset, z, m)

if __name__ == "__main__":
    main()
