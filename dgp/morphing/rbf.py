"""
Radial-basis-function (RBF) mesh morphing.

A displacement prescribed on the boundary nodes, delta_i = x_target_i - x_i, is extended to the whole domain as

    delta(x) = sum_i theta_i phi(||x - x_i|| / R),

where phi is a compactly supported radial kernel of support radius R and the coefficients theta solve the
Tikhonov-regularised system (Phi^T Phi + lam I) theta = Phi^T delta, with Phi_ij = phi(||x_j - x_i|| / R).
The analytic Jacobian determinant det(I + grad delta) is available at any point, to check that the map is
one-to-one (det J > 0).

All computations are done in float64 (JAX with x64 enabled).
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# phi(xi) = (1 - xi)^n * p(xi) / p(0) for xi < 1, 0 otherwise (xi = r / R); entries are (n, coefficients of p, lowest first)
KERNELS = {
    # Compactly supported polynomial kernel used for AirfRANS. Not a Wendland function: phi'(0) = -11,
    # so the radial function has a cusp at each centre (C^0 there, smooth elsewhere).
    "compact_poly": (12, [1, 1, 6, 0, 40, 128]),
    # Wendland functions phi_{3,k} (positive definite in R^d, d <= 3); wendland_c2 is used for VKI-LS59
    "wendland_c0": (2, [1]),
    "wendland_c2": (4, [1, 4]),
    "wendland_c4": (6, [3, 18, 35]),
    "wendland_c6": (8, [1, 8, 25, 32]),
    "wendland_c8": (10, [5, 50, 210, 450, 429]),
    "wendland_c10": (12, [9, 108, 566, 1644, 2697, 2048]),
    # truncated powers (1 - xi)^p (positive definite in 2D and 3D for p >= 2), phi'(0) = -p; p = 5 is CTPS C0
    "trunc_p3": (3, [1]), "trunc_p4": (4, [1]), "trunc_p5": (5, [1]), "trunc_p6": (6, [1]), "trunc_p8": (8, [1]),
}


def kernel(name):
    """Returns phi(xi) and its derivative dphi/dxi as JAX functions."""
    n, c = KERNELS[name]
    c = np.asarray(c, float) / c[0]
    dc = np.polynomial.polynomial.polyder(c)
    c_j, dc_j = jnp.asarray(c[::-1]), jnp.asarray(dc[::-1])

    def phi(xi):
        return jnp.where(xi < 1.0, (1.0 - xi) ** n * jnp.polyval(c_j, xi), 0.0)

    def dphi(xi):
        return jnp.where(xi < 1.0, -n * (1.0 - xi) ** (n - 1) * jnp.polyval(c_j, xi)
                         + (1.0 - xi) ** n * jnp.polyval(dc_j, xi), 0.0)
    return phi, dphi


def fit_rbf(Xb, Xt, kernel_name, radius, reg, solver="normal"):
    """Coefficients theta (n_ctrl, 2) of the displacement Xt - Xb at the control points Xb.

    solver = "normal": np.linalg.solve of the normal equations (Phi^T Phi + reg I) theta = Phi^T delta
                       (reg = 0: exact interpolation Phi theta = delta)
    solver = "eigh"  : same regularised solution through the eigendecomposition of the symmetric Phi,
                       theta = V diag(s / (s^2 + reg)) V^T delta, without squaring the condition number
    Returns theta and the largest residual |Phi theta - delta| at the control points.
    """
    phi, _ = kernel(kernel_name)
    Xb_j = jnp.asarray(Xb)
    Phi = phi(jnp.linalg.norm(Xb_j[:, None] - Xb_j[None], axis=-1) / radius)
    delta = Xt - Xb
    if solver == "eigh":
        s, V = jnp.linalg.eigh(Phi)
        theta = np.asarray(V @ ((s / (s ** 2 + reg))[:, None] * (V.T @ jnp.asarray(delta))))
        Phi = np.asarray(Phi)
    else:
        Phi = np.asarray(Phi)
        if reg > 0:
            theta = np.linalg.solve(Phi.T @ Phi + reg * np.eye(len(Phi)), Phi.T @ delta)
        else:
            theta = np.linalg.solve(Phi, delta)
    return theta, float(np.abs(Phi @ theta - delta).max())


@functools.partial(jax.jit, static_argnames=("kernel_name", "radius", "jacobian"))
def _eval_chunk(Xc, Xb, theta, kernel_name, radius, jacobian):
    phi, dphi = kernel(kernel_name)
    diff = Xc[:, None, :] - Xb[None]
    r = jnp.sqrt(jnp.sum(diff ** 2, -1) + 1e-30)
    disp = phi(r / radius) @ theta
    if not jacobian:
        return disp, None
    g = (dphi(r / radius) / radius / r)[..., None] * diff                # grad of phi_i at the points
    Jx = jnp.einsum("nbk,b->nk", g, theta[:, 0])
    Jy = jnp.einsum("nbk,b->nk", g, theta[:, 1])
    return disp, (1 + Jx[:, 0]) * (1 + Jy[:, 1]) - Jx[:, 1] * Jy[:, 0]


def eval_rbf(X, Xb, theta, kernel_name, radius, jacobian=False, chunk=4096):
    """Displacement at the points X (n, 2) and, if jacobian=True, det(I + grad delta) at the same points.
    The points are processed in chunks of fixed size, so the compiled kernel is reused."""
    n = len(X)
    Xp = np.vstack([X, np.zeros((-(-n // chunk) * chunk - n, 2))])
    Xb_j, th_j = jnp.asarray(Xb), jnp.asarray(theta)
    disp, det = [], []
    for i in range(0, len(Xp), chunk):
        d, j = _eval_chunk(jnp.asarray(Xp[i:i + chunk]), Xb_j, th_j, kernel_name, radius, jacobian)
        disp.append(np.asarray(d))
        if jacobian:
            det.append(np.asarray(j))
    disp = np.concatenate(disp)[:n]
    return (disp, np.concatenate(det)[:n]) if jacobian else disp
