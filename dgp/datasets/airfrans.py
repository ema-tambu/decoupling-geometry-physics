"""
Information-preserving transformation T = T_bal o T_geom for AirfRANS (Bonnet et al., NeurIPS 2022).

For one simulation:
  1. read the mesh nodes and fields (u, v, p) from <name>_internal.vtu, keep the box [-2, 4] x [-1.5, 1.5] and
     align the airfoil so that its leading edge is at (0, 0) and its trailing edge at (1, 0);
  2. split the airfoil surface loop (connectivity of <name>_aerofoil.vtp) into suction and pressure sides;
  3. boundary targets: each wall node keeps its chord-normalised abscissa and is projected vertically onto the
     circle of radius 0.5 centred at (0.5, 0); the outer rectangle is mapped onto the concentric circle through
     its corners (radius 3 sqrt(5) / 2);
  4. RBF morphing of the interior (compact polynomial kernel with a cusp, R = 10, lambda = 1e-6);
  5. polar coordinates around (0.5, 0) and radial wall correction: along each ray the morphed wall is moved
     exactly onto r = 0.5 by a strictly increasing affine map (this is T_geom);
  6. T_bal: eCDF of the radial coordinate pooled over the training meshes (see scripts/transform_airfrans.py).
Every node keeps its identity, so predictions in the transformed domain are mapped back without interpolation.
"""
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from dgp.morphing import eval_rbf, fit_rbf

BOUNDS = (-2.0, 4.0, -1.5, 1.5)                 # box kept around the airfoil
CENTER = np.array([0.5, 0.0])                   # centre of both target circles
R_WALL = 0.5
R_OUT = 3.0 * np.sqrt(5.0) / 2.0                # circle through the corners of the box
SPLITS = ("full_train", "scarce_train", "full_test", "reynolds_train", "reynolds_test", "aoa_train", "aoa_test")


@dataclass(frozen=True)
class MorphConfig:
    kernel: str = "compact_poly"
    radius: float = 10.0
    reg: float = 1e-6            # lambda in (Phi^T Phi + lambda I) theta = Phi^T delta
    n_outer: int = 1000          # points per side of the outer rectangle (corners included)
    chunk: int = 4000            # evaluation chunk size (memory)


# ---------------------------------------------------------------------------------------------
# reading the raw dataset
# ---------------------------------------------------------------------------------------------
def manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


def input_params(name):
    """[U_inf, alpha, NACA parameters] from the simulation name; 4-digit airfoils get -1 as third NACA parameter."""
    p = name.split("_")
    naca = [float(x) for x in p[4:]]
    if len(naca) < 4:
        naca.insert(2, -1.0)
    return [float(p[2]), float(p[3])] + naca


def read_simulation(data_dir, name):
    internal = pv.read(os.path.join(data_dir, name, name + "_internal.vtu"))
    aerofoil = pv.read(os.path.join(data_dir, name, name + "_aerofoil.vtp"))
    X_int = np.column_stack((internal.points[:, 0], internal.points[:, 1]))
    X_aero = np.column_stack((aerofoil.points[:, 0], aerofoil.points[:, 1]))
    fields = np.column_stack((internal.point_data["U"][:, 0], internal.point_data["U"][:, 1], internal.point_data["p"]))
    return X_int, X_aero, fields, internal, aerofoil


def cut_rectangle_bounds(X, fields, bounds):
    x_min, x_max, y_min, y_max = bounds
    mask = (X[:, 0] >= x_min) & (X[:, 0] <= x_max) & (X[:, 1] >= y_min) & (X[:, 1] <= y_max)
    return X[mask], fields[mask], mask


def leading_edge(X_int, X_aero):
    """Rotation about the trailing edge (1, 0) and shrinking in x, so that the leading edge is exactly at (0, 0)."""
    idx_min = np.argmin(X_aero[:, 0])
    idx_max = np.argmax(X_aero[:, 0])
    leading = np.array([X_aero[idx_min, 0], X_aero[idx_min, 1]])
    trailing = np.array([X_aero[idx_max, 0], X_aero[idx_max, 1]])
    if trailing[0] != 1.0 or trailing[1] != 0.0:
        raise ValueError("Trailing edge is not at (1.0, 0.0)")

    x_lead, y_lead = leading
    xi = lambda x: (1 - x)                      # noqa: E731
    alpha = np.arctan(y_lead / xi(x_lead))
    rho = (xi(x_lead) ** 2 + y_lead ** 2) ** 0.5
    theta = np.arctan(y_lead / xi(x_lead)) - alpha
    x_lead = xi(rho * np.cos(theta))
    y_lead = rho * np.sin(theta)
    if y_lead != 0.0:
        raise ValueError("y_lead is not 0.0 after rotation")

    for i in range(len(X_aero)):
        if X_aero[i, 0] != 1.0 and X_aero[i, 1] != 0:
            rho = (xi(X_aero[i, 0]) ** 2 + X_aero[i, 1] ** 2) ** 0.5
            theta = np.arctan(X_aero[i, 1] / xi(X_aero[i, 0])) - alpha
            x_i = xi(rho * np.cos(theta))
            X_aero[i, 0] = 1 - (1 - x_i) / (1 - x_lead)
            X_aero[i, 1] = rho * np.sin(theta)

    # interior points: rotation for all of them, x-shrinking only on the left of the trailing edge
    for i in range(len(X_int)):
        if X_int[i, 0] != 1.0 and X_int[i, 1] != 0:
            rho = (xi(X_int[i, 0]) ** 2 + X_int[i, 1] ** 2) ** 0.5
            if xi(X_int[i, 0]) < 0 and X_int[i, 1] > 0:
                theta = np.arctan(X_int[i, 1] / xi(X_int[i, 0])) - alpha + np.pi
            elif xi(X_int[i, 0]) < 0 and X_int[i, 1] < 0:
                theta = np.arctan(X_int[i, 1] / xi(X_int[i, 0])) - alpha - np.pi
            else:
                theta = np.arctan(X_int[i, 1] / xi(X_int[i, 0])) - alpha
            X_int[i, 0] = xi(rho * np.cos(theta))
            X_int[i, 1] = rho * np.sin(theta)
        if X_int[i, 0] < trailing[0]:
            X_int[i, 0] = 1 - (1 - X_int[i, 0]) / (1 - x_lead)

    if X_aero[idx_min, 0] != 0.0 or X_aero[idx_min, 1] != 0.0:
        raise ValueError("Leading edge is not at (0.0, 0.0) after the alignment")
    return X_int, X_aero


def load_sample(data_dir, name, with_cells=False):
    """Mesh after the box cut and the LE/TE alignment, the wall mask and the ordered suction/pressure sides."""
    X_int0, X_aero0, fields0, internal, aerofoil = read_simulation(data_dir, name)
    X_int, fields, keep = cut_rectangle_bounds(X_int0, fields0, BOUNDS)
    # wall nodes = mesh nodes coinciding with airfoil points (matched before the alignment)
    wall = cKDTree(X_aero0).query(X_int)[0] < 1e-12
    X_int, X_aero = leading_edge(X_int, X_aero0)
    X_int, X_aero = np.asarray(X_int, float), np.asarray(X_aero, float)

    loop = _ordered_loop(aerofoil.lines.reshape(-1, 3)[:, 1:], start=int(np.argmin(X_aero[:, 0])))
    suction, pressure = split_airfoil(X_aero[loop])
    out = dict(name=name, params=np.asarray(input_params(name)), X=X_int, fields=np.asarray(fields),
               wall=wall, suction=suction, pressure=pressure)
    if with_cells:
        new_idx = -np.ones(len(keep), dtype=np.int64)
        new_idx[keep] = np.arange(int(keep.sum()))
        out["cells"] = [m[(m >= 0).all(axis=1)] for m in (new_idx[c] for c in internal.cells_dict.values())]
    return out


def cell_centroids(sample):
    return np.vstack([sample["X"][c].mean(axis=1) for c in sample["cells"]])


def _ordered_loop(lines, start):
    nbr = {}
    for a, b in lines:
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    loop, prev = [start], None
    while True:
        nxt = [n for n in nbr[loop[-1]] if n != prev][0]
        if nxt == start:
            return np.array(loop)
        prev = loop[-1]
        loop.append(nxt)


def split_airfoil(loop_pts):
    """Closed loop -> (suction, pressure), each ordered LE -> TE and including both endpoints.
    LE = min-x point, TE = point closest to (1, 0); the side with larger mean y over 0.2 < x < 0.8 is the suction side."""
    loop_pts = np.roll(loop_pts, -int(np.argmin(loop_pts[:, 0])), axis=0)
    te = int(np.argmin(np.linalg.norm(loop_pts - [1.0, 0.0], axis=1)))
    a = loop_pts[:te + 1]
    b = np.vstack([loop_pts[te:], loop_pts[:1]])[::-1]
    xs = np.linspace(0.2, 0.8, 50)
    mean_y = [np.interp(xs, *s[np.argsort(s[:, 0])].T).mean() for s in (a, b)]
    return (a, b) if mean_y[0] > mean_y[1] else (b, a)


# ---------------------------------------------------------------------------------------------
# boundary targets and T_geom
# ---------------------------------------------------------------------------------------------
def airfoil_targets(suction, pressure):
    """Keep the chord-normalised x and project vertically onto the circle of radius R_WALL."""
    x0 = min(suction[:, 0].min(), pressure[:, 0].min())
    out = []
    for side, sign in ((suction, 1.0), (pressure, -1.0)):
        xn = (side[:, 0] - x0) / (1.0 - x0)
        y = sign * np.sqrt(np.clip(R_WALL ** 2 - (xn - CENTER[0]) ** 2, 0.0, None))
        out.append(np.column_stack([xn, y]))
    return out


def outer_targets(n):
    """Outer rectangle -> circle of radius R_OUT, concentric with the airfoil circle."""
    x0, x1, y0, y1 = BOUNDS
    t = np.linspace(0.0, 1.0, n)[1:-1]
    xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    xc = 1.0                                     # centre of the rectangle; the circle is shifted by -0.5
    src = [np.column_stack([xs, np.full_like(xs, y0)]), np.column_stack([xs, np.full_like(xs, y1)]),
           np.column_stack([np.full_like(ys, x0), ys]), np.column_stack([np.full_like(ys, x1), ys])]
    dst = [np.column_stack([xs - 0.5, -np.sqrt(R_OUT ** 2 - (xs - xc) ** 2)]),
           np.column_stack([xs - 0.5, np.sqrt(R_OUT ** 2 - (xs - xc) ** 2)]),
           np.column_stack([xc - np.sqrt(R_OUT ** 2 - ys ** 2) - 0.5, ys]),
           np.column_stack([xc + np.sqrt(R_OUT ** 2 - ys ** 2) - 0.5, ys])]
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return np.vstack(src + [corners]), np.vstack(dst + [corners - [0.5, 0.0]])


def control_points(sample, cfg):
    """Boundary nodes (sources) and their targets, without duplicates."""
    so, do = outer_targets(cfg.n_outer)
    ts, tp = airfoil_targets(sample["suction"], sample["pressure"])
    Xb = np.vstack([so, sample["suction"], sample["pressure"][1:-1]])   # LE and TE once (from the suction side)
    Xt = np.vstack([do, ts, tp[1:-1]])
    _, keep = np.unique(np.round(Xb, 13), axis=0, return_index=True)
    keep = np.sort(keep)
    return Xb[keep], Xt[keep]


def to_polar(X):
    d = X - CENTER
    return np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])


def wall_snap(theta, r, wall):
    """Move the morphed wall r_w(theta) exactly onto R_WALL with a strictly increasing affine map per ray,
       r_hat = R_WALL + (r - r_w(theta)) (R_OUT - R_WALL) / (R_OUT - r_w(theta)),
    with r_w the periodic piecewise-linear interpolant of the wall nodes in theta."""
    tw, rw = theta[wall], r[wall]
    o = np.argsort(tw)
    tw, rw = tw[o], rw[o]
    tw = np.r_[tw[-1] - 2 * np.pi, tw, tw[0] + 2 * np.pi]
    rw = np.r_[rw[-1], rw, rw[0]]
    r_w = np.interp(theta, tw, rw)
    r_hat = R_WALL + (r - r_w) * (R_OUT - R_WALL) / (R_OUT - r_w)
    r_hat[wall] = R_WALL
    return r_hat, r_w


def polar_rotated(theta):
    """Angle rotated by pi: 0 points upstream and the periodic seam lies on the wake line behind the trailing edge."""
    return (theta + 2 * np.pi) % (2 * np.pi) - np.pi


def transform_geometry(sample, cfg=MorphConfig(), jacobian=False):
    """T_geom for one sample: coordinates after the RBF morphing (X_rbf) and after the wall correction (X_Tgeom),
    the polar coordinates (theta rotated by pi, r) used by T_bal, det J at the nodes and diagnostics."""
    Xb, Xt = control_points(sample, cfg)
    theta_rbf, ctrl_res = fit_rbf(Xb, Xt, cfg.kernel, cfg.radius, cfg.reg, solver="normal")
    X = sample["X"]
    out = eval_rbf(X, Xb, theta_rbf, cfg.kernel, cfg.radius, jacobian=jacobian, chunk=cfg.chunk)
    disp, det = out if jacobian else (out, None)
    X_rbf = X + disp
    th, r = to_polar(X_rbf)
    r_hat, _ = wall_snap(th, r, sample["wall"])
    X_Tgeom = CENTER + np.column_stack([r_hat * np.cos(th), r_hat * np.sin(th)])

    wall = sample["wall"]
    diag = dict(n_nodes=len(X), n_ctrl=len(Xb), ctrl_res=ctrl_res,
                wall_res=float(np.abs(r[wall] - R_WALL).max()),
                inside_before_snap=int(((r < R_WALL) & ~wall).sum()),
                below_wall_after_snap=int(((r_hat < R_WALL) & ~wall).sum()),
                max_snap_shift=float(np.abs(r_hat - r).max()))
    if det is not None:
        diag.update(min_det=float(det.min()), n_folded=int((det <= 0).sum()))
    if jacobian and "cells" in sample:
        _, det_c = eval_rbf(cell_centroids(sample), Xb, theta_rbf, cfg.kernel, cfg.radius, jacobian=True, chunk=cfg.chunk)
        diag.update(min_det_centroids=float(det_c.min()), n_folded_centroids=int((det_c <= 0).sum()))
    return dict(X_rbf=X_rbf, X_Tgeom=X_Tgeom, theta=polar_rotated(th), r=r_hat, det=det, diag=diag, config=asdict(cfg))
