# GINO baseline: wrapper around neuralop.models.GINO (https://github.com/neuraloperator/neuraloperator, MIT License)
# for one 2-D mesh of variable size per batch, input [parameters, x, y]; the global parameters condition the FNO
# through adaptive instance normalisation (ada_in).
import torch
import torch.nn as nn
from neuralop.models import GINO

class UnifiedGINOWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dataset_name = config['data'].get('dataset_name', 'airfrans')
        
        # Determine feature dimensions based on the new standardized dataset format
        if self.dataset_name == "vki_ls59":
            in_channels = 12
            # 9 global parameters (POD + angle + mach), excluding the local SDF
            ada_in_features = 9 
        else:
            # AirfRANS
            in_channels = 8
            ada_in_features = 6
            
        out_channels = config['model']['out_dim']
        
        self.grid_res_x = int(config['model']['grid_resolution_x_gino'])
        self.grid_res_y = int(config['model']['grid_resolution_y_gino'])
        
        # Handle list conversion for modes dynamically
        modes = config['model'].get('fno_n_modes_gino', 16)
        if isinstance(modes, list):
            fno_modes = tuple(modes)
        else:
            fno_modes = (int(modes), int(modes))
        
        self.gino = GINO(
            in_channels=in_channels,           
            out_channels=out_channels,          
            gno_coord_dim=2,         
            gno_use_open3d=False,
            
            in_gno_radius=float(config['model']['in_gno_radius_gino']),
            out_gno_radius=float(config['model']['out_gno_radius_gino']),
            
            fno_in_channels=in_channels,
            fno_n_modes=fno_modes,
            fno_hidden_channels=int(config['model']['fno_hidden_channels_gino']),
            fno_n_layers=int(config['model']['fno_n_layers_gino']),
            
            fno_norm="ada_in",
            fno_ada_in_features=ada_in_features,   
            fno_ada_in_dim=ada_in_features         
        )

    def forward(self, batch_x):
        """
        Expects a batch of shape (1, N_nodes, features)
        Coordinates are guaranteed to be the last 2 columns by dataset.py
        """
        clean_x = batch_x
        geom_in = batch_x[:, :, -2:] 
        
        if self.dataset_name == "vki_ls59":
            # Extract first 9 global scalars, skip index 9 (SDF)
            ada_in = batch_x[:, 0, 0:9]
        else:
            # AirfRANS: Extract first 6 global scalars
            ada_in = batch_x[:, 0, 0:6]
        
        # Dynamic Bounding Box Construction
        x_min, x_max = geom_in[:, :, 0].min(), geom_in[:, :, 0].max()
        y_min, y_max = geom_in[:, :, 1].min(), geom_in[:, :, 1].max()
        
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        
        x_coords = torch.linspace(x_min - x_pad, x_max + x_pad, self.grid_res_x, device=batch_x.device)
        y_coords = torch.linspace(y_min - y_pad, y_max + y_pad, self.grid_res_y, device=batch_x.device)
        
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        latent_queries = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        pred = self.gino(
            input_geom=geom_in,
            latent_queries=latent_queries,
            output_queries=geom_in,
            x=clean_x,
            ada_in=ada_in
        )
        
        return pred