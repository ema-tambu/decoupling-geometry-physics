"""
Information-preserving transformation T = T_bal o T_geom for VKI-LS59 (PLAID benchmark, Casenave et al. 2025).

All meshes share a structured 121 x 301 grid (node = j * 121 + i). Bottom boundary = line i = 0 (periodic line
j < 60, suction wall j = 60..181, periodic line j > 181); top boundary = line i = 120 (same with the pressure wall);
inflow = row j = 0; outflow = row j = 300. The reference domain is the rectangle [-0.5, 2] x [0, 1].

T_geom for one mesh:
  1. boundary targets: inflow -> x = -0.5, outflow -> x = 2, bottom line -> y = 0, top line -> y = 1. On each blade
     wall the target abscissa is the normalised arc length along the wall, mapped onto [x_LE, x_TE] of that wall
     (strictly increasing, also where the wall curves back near the rounded trailing edge); periodic lines keep x;
  2. RBF morphing of the interior (Wendland C2, R = 2, lambda = 1e-10, solved through the eigendecomposition of Phi);
  3. boundary correction, two strictly increasing 1-D affine maps that put the morphed boundaries exactly on the
     rectangle:
       x_hat = -0.5 + 2.5 (x - x_in(y)) / (x_out(y) - x_in(y))
       y_hat = (y - y_bot(x_hat)) / (y_top(x_hat) - y_bot(x_hat))
T_bal: eCDF of each coordinate pooled over the training meshes (see scripts/transform_vki_ls59.py).
"""
import numpy as np

from dgp.morphing import eval_rbf, fit_rbf

NI, NJ = 121, 301
JW = np.arange(60, 182)                    # wall rows (suction side on i = 0, pressure side on i = 120)
X0, X1, Y0, Y1 = -0.5, 2.0, 0.0, 1.0
KERNEL, R_RBF, LAM = "wendland_c2", 2.0, 1e-10

BOT = np.arange(NJ) * NI                   # line i = 0
TOP = BOT + NI - 1                         # line i = 120
INF = np.arange(NI)                        # row j = 0
OUT = (NJ - 1) * NI + np.arange(NI)        # row j = 300


# ------------------------------------------------------------------------------------------------ PLAID
def load_plaid(processes=4):
    """VKI-LS59 from the Hugging Face hub (PLAID-datasets/VKI-LS59) as a PLAID dataset."""
    from datasets import load_dataset
    from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid
    hf = load_dataset("PLAID-datasets/VKI-LS59", split="all_samples")
    return huggingface_dataset_to_plaid(hf, processes_number=processes)


def sample_arrays(sample, with_fields=True):
    """Node coordinates, nodal tags, quads, SDF and (optionally) Mach / nu_t fields and scalars of one PLAID sample."""
    base = "Base_2_2"
    out = dict(X=sample.get_nodes(base_name=base)[:, :2].astype(float),
               tags={k: np.asarray(v) for k, v in sample.get_nodal_tags(base_name=base).items()},
               quads=sample.get_elements(base_name=base)["QUAD_4"],
               sdf=sample.get_field("sdf", base_name=base))
    if with_fields:
        out["fields"] = np.column_stack((sample.get_field("mach", base_name=base), sample.get_field("nut", base_name=base)))
        out["angle_in"], out["mach_out"] = sample.get_scalar("angle_in"), sample.get_scalar("mach_out")
    return out


def check_structure(tags):
    """Asserts the structured layout assumed here, from the PLAID nodal tags."""
    j = np.arange(NJ)
    ok = (np.array_equal(tags["Inflow"], np.arange(NI)) and np.array_equal(tags["Outflow"], (NJ - 1) * NI + np.arange(NI))
          and np.array_equal(tags["Extrado"], JW * NI) and np.array_equal(tags["Intrado"], JW * NI + NI - 1)
          and set(tags["Periodic_1"]) == set(np.r_[j[:60] * NI, j[:60] * NI + NI - 1])
          and set(tags["Periodic_2"]) == set(np.r_[j[182:] * NI, j[182:] * NI + NI - 1]))
    if not ok:
        raise ValueError("mesh does not have the expected 121 x 301 structured layout")


# ------------------------------------------------------------------------------------------------ T_geom
def wall_line_x_targets(L):
    """L: (NJ, 2) nodes of the bottom or top line. Target x of every node of the line (arc length on the wall)."""
    xt = L[:, 0].copy()
    w = L[JW]
    inc = np.sqrt(np.diff(w[:, 0]) ** 2 + np.diff(w[:, 1]) ** 2)
    c = np.r_[0.0, np.cumsum(inc)]
    xt[JW] = w[0, 0] + (w[-1, 0] - w[0, 0]) * c / c[-1]
    xt[0], xt[-1] = X0, X1
    return xt


def control_points(X):
    """Each boundary node exactly once (840). Returns node indices, sources and targets."""
    idx = np.r_[BOT, TOP, INF[1:-1], OUT[1:-1]]
    t = np.arange(1, NI - 1) / (NI - 1)
    Xt = np.vstack([np.column_stack([wall_line_x_targets(X[BOT]), np.full(NJ, Y0)]),
                    np.column_stack([wall_line_x_targets(X[TOP]), np.full(NJ, Y1)]),
                    np.column_stack([np.full(NI - 2, X0), t]),
                    np.column_stack([np.full(NI - 2, X1), t])])
    return idx, X[idx], Xt


def boundary_correction(Xm):
    """The two affine 1-D corrections (module docstring). Returns X_Tgeom and diagnostics; diag['monotone'] is True
    when all denominators and graph conditions are positive, i.e. the correction is strictly increasing."""
    x, y = Xm[:, 0].copy(), Xm[:, 1].copy()
    d = {}
    yin, xin, yout, xout = y[INF], x[INF], y[OUT], x[OUT]
    d["inflow_graph_min_dy"] = float(np.diff(yin).min())
    d["outflow_graph_min_dy"] = float(np.diff(yout).min())
    xl, xr = np.interp(y, yin, xin), np.interp(y, yout, xout)
    d["x_denom_min"] = float((xr - xl).min())
    xh = X0 + (x - xl) * (X1 - X0) / (xr - xl)
    xh[INF], xh[OUT] = X0, X1
    d["bot_graph_min_dx"] = float(np.diff(xh[BOT]).min())
    d["top_graph_min_dx"] = float(np.diff(xh[TOP]).min())
    yb, yt = np.interp(xh, xh[BOT], y[BOT]), np.interp(xh, xh[TOP], y[TOP])
    d["y_denom_min"] = float((yt - yb).min())
    yh = Y0 + (y - yb) * (Y1 - Y0) / (yt - yb)
    yh[BOT], yh[TOP] = Y0, Y1
    d["max_shift_x"] = float(np.abs(xh - x).max())
    d["max_shift_y"] = float(np.abs(yh - y).max())
    d["monotone"] = bool(min(d["inflow_graph_min_dy"], d["outflow_graph_min_dy"], d["x_denom_min"],
                             d["bot_graph_min_dx"], d["top_graph_min_dx"], d["y_denom_min"]) > 0)
    return np.column_stack([xh, yh]), d


def outside_box(X, tol=0.0):
    return int(((X[:, 0] < X0 - tol) | (X[:, 0] > X1 + tol) | (X[:, 1] < Y0 - tol) | (X[:, 1] > Y1 + tol)).sum())


def transform_geometry(X, jacobian=False, quads=None, radius=R_RBF, lam=LAM):
    """T_geom for one mesh (X: original node coordinates). With jacobian=True, det J of the RBF map at the nodes
    and, if quads are given, at the quad centroids."""
    idx, Xb, Xt = control_points(X)
    th, ctrl_res = fit_rbf(Xb, Xt, KERNEL, radius, lam, solver="eigh")
    out = {}
    if jacobian:
        disp, out["det_nodes"] = eval_rbf(X, Xb, th, KERNEL, radius, jacobian=True, chunk=8192)
        if quads is not None:
            out["det_cent"] = eval_rbf(X[quads].mean(1), Xb, th, KERNEL, radius, jacobian=True, chunk=8192)[1]
    else:
        disp = eval_rbf(X, Xb, th, KERNEL, radius, chunk=8192)
    Xm = X + disp
    Xg, diag = boundary_correction(Xm)
    diag["ctrl_res"] = ctrl_res
    diag["wall_res_before"] = float(np.r_[np.abs(Xm[BOT[JW], 1] - Y0), np.abs(Xm[TOP[JW], 1] - Y1)].max())
    diag["outside_box_before"], diag["outside_box_after"] = outside_box(Xm), outside_box(Xg)
    out.update(X_rbf=Xm, X_Tgeom=Xg, diag=diag)
    return out


def quad_area(X, quads):
    P = X[quads]
    return 0.5 * np.sum(P[..., 0] * np.roll(P[..., 1], -1, 1) - np.roll(P[..., 0], -1, 1) * P[..., 1], axis=1)


def apply_tbal(Xg, ecdf_x, ecdf_y):
    return np.column_stack([ecdf_x(Xg[:, 0]), ecdf_y(Xg[:, 1])])
