import torch

class PhysicalMetrics:
    """
    Computes physical evaluation metrics. 
    Expects inputs in the normalized space, un-normalizes them internally 
    using the provided config stats, and computes metrics on the GPU.
    """
    def __init__(self, config, device):
        # Build the dynamic keys based on the dataset and target channel
        dataset_name = config['data'].get('dataset_name', 'airfrans')
        dataset_version = config['data'].get('dataset_version', 'full')
        target_channel = config['data']['target_channel']
        channels = list(target_channel) if isinstance(target_channel, (list, tuple)) else [target_channel]
        self.multi = len(channels) > 1
        
        norm_key = f"airfrans_{dataset_version}" if dataset_name == "airfrans" else "vki_ls59"
        
        # Move normalization stats to the correct device immediately
        out_mean = [float(config['normalization'][f'{norm_key}_target_{c}_mean']) for c in channels]
        out_std = [float(config['normalization'][f'{norm_key}_target_{c}_std']) for c in channels]
        
        # scalars for one channel (unchanged behaviour), one entry per channel otherwise
        self.mean = torch.tensor(out_mean if self.multi else out_mean[0], dtype=torch.float32, device=device)
        self.std = torch.tensor(out_std if self.multi else out_std[0], dtype=torch.float32, device=device)
        self.channels = channels

    def _unnormalize(self, tensor):
        """Brings the tensor back to physical (un-normed) space."""
        return (tensor * self.std) + self.mean

    def nrmse(self, target, pred):
        """Normalized Root Mean Square Error (Scalar)."""
        rmse = torch.sqrt(torch.mean((target - pred)**2))
        val_range = torch.max(target) - torch.min(target)
        return (rmse / (val_range + 1e-8))

    def relative_l2_error(self, target, pred):
        """
        Relative L2 Error. 
        Calculates: sqrt( sum((target - pred)^2) / sum(target^2) )
        """
        numerator = torch.sum((target - pred)**2)
        denominator = torch.sum(target**2)
        return torch.sqrt(numerator / (denominator + 1e-8))

    def evaluate(self, target_norm, pred_norm):
        """
        Main entry point. Pass the batched, normalized tensors.
        """
        with torch.no_grad():
            # 1. Compute in Normalized Space
            rel_l2_norm = self.relative_l2_error(target_norm, pred_norm)

            # 2. Un-normalize to Physical Space
            target_phys = self._unnormalize(target_norm)
            pred_phys = self._unnormalize(pred_norm)
            
            # 3. Compute in Physical Space
            metrics_dict = {
                "NRMSE_Phys": self.nrmse(target_phys, pred_phys),
                "Rel_L2_Phys": self.relative_l2_error(target_phys, pred_phys),
                "Rel_L2_Norm": rel_l2_norm
            }
            if self.multi:
                # the pooled metrics above mix fields with different ranges: report every field separately
                for j, c in enumerate(self.channels):
                    metrics_dict[f"NRMSE_Phys_ch{c}"] = self.nrmse(target_phys[:, j], pred_phys[:, j])
            
        return metrics_dict