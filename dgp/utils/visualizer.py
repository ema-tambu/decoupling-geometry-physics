import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_pointcloud_figure(coords, target, pred, max_edge_ratio=0.05):
    """
    Creates a 1x3 comparison plot using Delaunay Triangulation for CFD meshes.
    Includes a dynamic mask to remove artifact triangles in non-convex domains.
    """
    def to_np_1d(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().float().numpy()
        return np.squeeze(tensor)

    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().float().numpy()

    target = to_np_1d(target)
    pred = to_np_1d(pred)

    gt_range = np.max(target) - np.min(target)
    nrmse_field = np.abs(target - pred) / (gt_range + 1e-8)

    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Tighten horizontal space
    plt.subplots_adjust(wspace=0.1)
    
    x, y = coords[:, 0], coords[:, 1]
    vmn, vmx = np.min(target), np.max(target)

    # 1. Base Triangulation
    triang = tri.Triangulation(x, y)
    
    # 2. Masking large artifact triangles (Delaunay Convex Hull fix)
    # Extract the coordinates of the 3 points forming each triangle
    triangles = triang.triangles
    xt = x[triangles]
    yt = y[triangles]
    
    # Calculate the maximum edge length squared for each triangle
    # Edges: 0-1, 1-2, 2-0
    dx = np.column_stack((xt[:, 1] - xt[:, 0], xt[:, 2] - xt[:, 1], xt[:, 0] - xt[:, 2]))
    dy = np.column_stack((yt[:, 1] - yt[:, 0], yt[:, 2] - yt[:, 1], yt[:, 0] - yt[:, 2]))
    max_edge_len_sq = np.max(dx**2 + dy**2, axis=1)
    
    # Calculate domain diagonal squared
    domain_diag_sq = (np.max(x) - np.min(x))**2 + (np.max(y) - np.min(y))**2
    
    # Mask out triangles whose max edge is longer than a percentage of the domain diagonal
    # 0.05 (5%) is usually perfect for dense CFD meshes to cleanly cut out the airfoil hole
    mask = max_edge_len_sq > (domain_diag_sq * (max_edge_ratio**2))
    triang.set_mask(mask)

    shading_style = 'gouraud'

    # --- Plot 1: Ground Truth ---
    tc0 = axs[0].tripcolor(triang, target, cmap='turbo', vmin=vmn, vmax=vmx, shading=shading_style)
    axs[0].set_title('Ground Truth')
    
    # Phantom colorbar to force the exact same sizing as the others
    divider_gt = make_axes_locatable(axs[0])
    cax_gt = divider_gt.append_axes("right", size="5%", pad=0.15)
    cax_gt.axis('off')

    # --- Plot 2: Prediction ---
    tc1 = axs[1].tripcolor(triang, pred, cmap='turbo', vmin=vmn, vmax=vmx, shading=shading_style)
    axs[1].set_title('Prediction')

    divider0 = make_axes_locatable(axs[1])
    cax0 = divider0.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(tc0, cax=cax0)

    # --- Plot 3: NRMSE Field (Locked to 0.0 - 0.025) ---
    tc2 = axs[2].tripcolor(triang, nrmse_field, cmap='magma', vmin=0.0, vmax=0.025, shading=shading_style)
    axs[2].set_title('Point-wise NRMSE')
    
    divider1 = make_axes_locatable(axs[2])
    cax1 = divider1.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(tc2, cax=cax1)

    # Final cleanup: Remove borders entirely for a clean look
    for ax in axs:
        ax.set_aspect('equal', 'box')
        ax.axis('off') 
        
    return fig