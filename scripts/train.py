"""
Training of the INR (shift-modulated SIREN), GINO or Transolver++ on a transformed dataset.

usage (one GPU):     python scripts/train.py --config configs/airfrans/inr_tbal.yaml --data.target_channel 0
usage (several GPUs): torchrun --standalone --nproc_per_node 4 scripts/train.py --config ...
Any config entry can be overridden on the command line (--group.key value). Outputs go to runs/exp_<time>_<id>/
(best_model.pt on the test loss, training_log.txt, metrics.csv, TensorBoard logs, plots of test meshes).
"""
import os
import time
import random
from contextlib import nullcontext
import torch
import torch.nn as nn
import yaml
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import apex
    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

from dgp.utils.config_parser import load_config
from dgp.utils.distributed import setup_distributed, cleanup_distributed, distributed_average
from dgp.utils.logger import ExperimentLogger
from dgp.utils.metrics import PhysicalMetrics
from dgp.utils.visualizer import create_pointcloud_figure
from dgp.data.dataset import CFDPointcloudDataset
from dgp.models import build_model

def main():
    global_start_time = time.time()

    # Configuration and distributed setup
    config = load_config()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    is_main = (rank == 0)
    architecture = config['model']['architecture']
    torch.manual_seed(config['system']['seed'])
    
    # Kernel optimization flags for Ampere architectures
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger = ExperimentLogger(config, is_main_process=is_main)
    if is_main:
        with open(os.path.join(logger.exp_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    # In-memory dataset initialization
    data_path = config['data']['base_folder']
    train_dataset = CFDPointcloudDataset(data_path, config, is_train=True)
    val_dataset = CFDPointcloudDataset(data_path, config, is_train=False)

    # VRAM Caching
    if is_main: 
        print("Pushing entire dataset to GPU VRAM...")
    if architecture == "siren":
        train_dataset.inputs = train_dataset.inputs.to(device)
        train_dataset.targets = train_dataset.targets.to(device)
        val_dataset.inputs = val_dataset.inputs.to(device)
        val_dataset.targets = val_dataset.targets.to(device)
    else:
        # intact meshes are lists of tensors
        train_dataset.inputs = [m.to(device) for m in train_dataset.inputs]
        train_dataset.targets = [m.to(device) for m in train_dataset.targets]
        val_dataset.inputs = [m.to(device) for m in val_dataset.inputs]
        val_dataset.targets = [m.to(device) for m in val_dataset.targets]

    # Samplers: batches of point chunks for the INR, one intact mesh at a time for GINO and Transolver++
    if config['data']['shuffle']:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset)
    if distributed:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = SequentialSampler(val_dataset)

    if architecture == "siren":
        train_iterable = BatchSampler(
            train_sampler, 
            batch_size=config['system']['macro_batch_size'], 
            drop_last=False
        )
        val_bsz = config['system'].get('val_macro_batch_size', config['system']['macro_batch_size'])
        val_iterable = BatchSampler(
            val_sampler, 
            batch_size=val_bsz, 
            drop_last=False
        )
    else:
        train_iterable = train_sampler
        val_iterable = val_sampler

    model = build_model(config).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    cuda_graphs = False   # set below when compile_mode is reduce-overhead
    # Inductor compilation with intelligent overhead reduction
    if architecture != "gino":
        try:
            # CUDA graphs (compile_mode reduce-overhead) need fixed tensor sizes: possible for the INR point chunks
            compile_mode = config['system'].get('compile_mode', "default")
            cuda_graphs = compile_mode == "reduce-overhead"

            model = torch.compile(model, mode=compile_mode)
            if is_main: 
                print(f"torch.compile ({compile_mode}) enabled successfully.")
        except Exception as e:
            if is_main: 
                print(f"error while torch.compile; skipped: {e}")
    else:
        if is_main:
            print("torch.compile bypassed for GINO (incompatible with torch_scatter).")

    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Number of parameters: {num_params}")
        logger.log_model_params(num_params)

    # Optimizer Setup
    opt_choice = config['system'].get('optimizer', 'adamw').lower()
    lr = float(config['system']['learning_rate'])
    wd = float(config['system'].get('weight_decay', 1e-4))

    # GINO (FNO) uses complex64 weights, which are incompatible with fused PyTorch optimizers
    can_use_fused = False if architecture == "gino" else True
    
    if opt_choice == 'lamb' and HAVE_APEX:
        if architecture == "gino":
            if is_main:
                print("Warning: GINO uses complex weights. Bypassing FusedLAMB and using unfused AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=False)
        else:
            optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=lr, weight_decay=wd)
    else:
        if opt_choice == 'lamb':
            # a silent switch to AdamW trains the INR much worse than LAMB: refuse unless explicitly allowed
            if os.environ.get("ALLOW_LAMB_FALLBACK") != "1":
                raise RuntimeError("optimizer 'lamb' requested but NVIDIA apex is not installed (set ALLOW_LAMB_FALLBACK=1 to use AdamW)")
            if is_main:
                print("Warning: Apex not found, falling back to AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=can_use_fused)

    criterion = nn.MSELoss() 

    # LR Scheduler Setup
    scheduler_type = config['system'].get('lr_scheduler', 'none').lower()
    # one optimizer step every grad_accum_steps batches plus one at the end of each epoch (ceil, not floor)
    total_steps = -(-len(train_iterable) // config['system'].get('grad_accum_steps', 1)) * config['system']['num_epochs']
    
    if scheduler_type == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config['system']['learning_rate'], 
            total_steps=total_steps,
            pct_start=0.05
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    else:
        scheduler = None

    physical_metrics = PhysicalMetrics(config, device)
    # PyTorch FFTs (used by GINO) do not support bfloat16 yet.
    # SIRENs (INRs) suffer catastrophic precision loss in mixed precision
    if architecture in ["gino", "siren"]:
        amp_dtype = torch.float32
        use_amp = False
        if is_main:
            print(f"AMP (BFloat16) disabled for {architecture} to maintain numerical precision.")
    else:
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        use_amp = (device.type == "cuda")

    best_val_loss = float('inf')
    num_epochs = config['system']['num_epochs']
    grad_accum_steps = config['system'].get('grad_accum_steps', 1)

    if is_main: 
        print("\n" + "="*40)
        print(" EXPERIMENT HYPERPARAMETERS")
        print("="*40)
        print(f" Dataset        : {config['data'].get('dataset_name')} ({config['data'].get('dataset_version', 'full')})")
        print(f" Target Channel : {config['data'].get('target_channel')}")
        print(f" Architecture   : {config['model'].get('architecture')}")
        print(f" Epochs         : {config['system'].get('num_epochs')}")
        print(f" Learning Rate  : {config['system'].get('learning_rate')}")
        if opt_choice == 'lamb' and HAVE_APEX:
           print(f" Optimizer      : FusedLAMB")
        else:
            print(f" Optimizer      : AdamW")
        print(f" Scheduler      : {config['system'].get('lr_scheduler')}")
        
        if architecture == 'siren':
            macro = config['system'].get('macro_batch_size', 1)
            n_pts = config['model'].get('N', 8192)
            print(f" Macro Batch    : {macro}")
            print(f" Chunk Size (N) : {n_pts}")
            print(f" Effective Pts  : {macro * n_pts} per forward pass")
            
        if config['model'].get('architecture') == 'siren':
            print(f" Omega          : {config['model'].get('omega_siren')}")
            print(f" Omega_0        : {config['model'].get('omega_0_siren')}")
            print(f" Hidden Dim     : {config['model'].get('hidden_dim_siren')}")
            print(f" Layers         : {config['model'].get('num_hidden_layers_siren')}")
        print("="*40 + "\n")

        print("\n\nStarting training loop...")

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # Dynamic fast VRAM shuffling for point-wise models
        if architecture == "siren":
            train_dataset.reshuffle()

        # Training Phase
        model.train()
        train_loss_sum = torch.tensor(0.0, device=device)  
        train_numel = 0  
        
        if opt_choice == 'lamb' and HAVE_APEX:
            optimizer.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)

        for step, batch_indices in enumerate(train_iterable):
            
            # Fetch Data
            if architecture == "siren":
                # the INR takes a 2-D tensor (points, features)
                idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
                batch_x = train_dataset.inputs[idx_tensor].view(-1, train_dataset.inputs.shape[-1])
                batch_y = train_dataset.targets[idx_tensor].view(-1, train_dataset.targets.shape[-1])
            else:
                # one intact mesh, with a batch dimension of 1
                batch_x = train_dataset.inputs[batch_indices].unsqueeze(0)
                batch_y = train_dataset.targets[batch_indices].unsqueeze(0)
            
            # Gradient Accumulation
            # During gradient accumulation steps, we use no_sync() to prevent DDP from synchronizing gradients
            # across GPUs prematurely. We only synchronize and step the optimizer at the end of the accumulation cycle.
            is_accumulating = (step + 1) % grad_accum_steps != 0 and (step + 1) != len(train_iterable)
            sync_context = model.no_sync() if (distributed and is_accumulating) else nullcontext()
            
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if cuda_graphs:
                        torch.compiler.cudagraph_mark_step_begin()   # each step replays the graph: outputs of the previous one are released
                    pred_y = model(batch_x)
                    loss = criterion(pred_y, batch_y)
                    # Scale loss by accumulation steps
                    scaled_loss = loss / grad_accum_steps
                    
                scaled_loss.backward()
            
            # Optimizer Step
            if not is_accumulating:
                if config['system'].get('grad_clip'):          # optional global gradient-norm clipping (off by default)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['system']['grad_clip']))
                optimizer.step()
                if opt_choice == 'lamb' and HAVE_APEX:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)

                if scheduler is not None and scheduler_type == 'one_cycle':
                    scheduler.step()

            bsz = batch_x.size(0)
            train_loss_sum += loss.detach().float() * bsz
            train_numel += bsz

        train_loss = distributed_average(train_loss_sum, train_numel, device, world_size)

        # Validation Phase
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)  
        val_numel = 0  
        
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_indices in val_iterable:
                if architecture == "siren":
                    idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
                    batch_x = val_dataset.inputs[idx_tensor].reshape(-1, val_dataset.inputs.shape[-1])
                    batch_y = val_dataset.targets[idx_tensor].reshape(-1, val_dataset.targets.shape[-1])
                else:
                    batch_x = val_dataset.inputs[batch_indices].unsqueeze(0)
                    batch_y = val_dataset.targets[batch_indices].unsqueeze(0)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if cuda_graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    pred_y = model(batch_x)
                    loss = criterion(pred_y, batch_y)

                bsz = batch_x.size(0)
                val_loss_sum += loss.detach().float() * bsz
                val_numel += bsz
                
                # Flatten the batch and node dimensions to (Total_Nodes, Features) 
                # so meshes of varying sizes can be concatenated vertically
                val_preds_list.append(pred_y.detach().reshape(-1, pred_y.shape[-1]).clone())   # clone: CUDA-graph outputs are overwritten by the next replay
                val_targets_list.append(batch_y.detach().reshape(-1, batch_y.shape[-1]))

        val_loss = distributed_average(val_loss_sum, val_numel, device, world_size)

        # Compute metrics globally to avoid local division-by-zero
        all_preds = torch.cat(val_preds_list, dim=0)
        all_targets = torch.cat(val_targets_list, dim=0)
        epoch_metrics = physical_metrics.evaluate(all_targets, all_preds)
        
        # Average the final global metrics across DDP ranks
        for k in epoch_metrics:
            # Ensure it's a tensor before reducing
            if not isinstance(epoch_metrics[k], torch.Tensor):
                epoch_metrics[k] = torch.tensor(epoch_metrics[k], device=device)
            epoch_metrics[k] = distributed_average(epoch_metrics[k], 1, device, world_size)

        is_new_best = val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss
            if is_main:
                save_path = os.path.join(logger.exp_dir, "best_model.pt")
                model_to_save = model.module if distributed else model
                torch.save(model_to_save.state_dict(), save_path)

        epoch_duration = time.time() - epoch_start_time
        if is_main:
            logger.log_metrics(epoch, train_loss, val_loss, epoch_duration, epoch_metrics, is_new_best)

        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_type == 'cosine':
                scheduler.step()

    # =========== END OF TRAINING LOOP ===========

    # Post-training inference evaluation and visualization rendering
    # (only on main process)
    if is_main:
        print("\n" + "="*50)
        print("Training complete. Generating final evaluation plots...")
        
        best_model_path = os.path.join(logger.exp_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model_to_eval = model.module if distributed else model
            # Robust loading for external libraries (GINO) requires a bit more care
            state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
            # Remove library-injected metadata keys that confuse PyTorch
            if "_metadata" in state_dict:
                del state_dict["_metadata"]
            # strict=False allows the wrapper to ignore any leftover non-weight artifacts
            model_to_eval.load_state_dict(state_dict, strict=False)

            model_to_eval.eval()

            total_meshes = len(val_dataset.mesh_lengths)
            num_plot_samples = config['system'].get('num_plot_samples', 1)
            num_plots = min(num_plot_samples, total_meshes)
            random_indices = random.sample(range(total_meshes), num_plots)

            with torch.no_grad():
                for i, mesh_idx in enumerate(random_indices):
                    try:
                        mesh_x_norm, mesh_y_norm = val_dataset.get_intact_mesh(mesh_idx=mesh_idx)
                        mesh_x_norm = mesh_x_norm.to(device)
                        
                        if architecture in ["gino", "transolver"]:
                            mesh_x_seqs = mesh_x_norm.unsqueeze(0)
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                pred_seqs = model_to_eval(mesh_x_seqs)
                            pred_y_flat = pred_seqs.squeeze(0)
                            
                        elif architecture == "siren":
                            # SIREN evaluates points individually, no padding or unsqueezing required
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                pred_y_flat = model_to_eval(mesh_x_norm)

                        # Un-normalize predictions
                        pred_phys = physical_metrics._unnormalize(pred_y_flat)
                        target_phys = physical_metrics._unnormalize(mesh_y_norm.to(device))

                        # Extract plotting coordinates (Now works uniformly for ALL datasets!)
                        xy_coords = val_dataset.get_original_coords(mesh_idx)
                        if xy_coords is not None:
                            xy_coords = xy_coords.to(device)
                        else:
                            raise ValueError("Could not find original validation coordinates in the dataset.")
                        
                        fig = create_pointcloud_figure(xy_coords, target_phys, pred_phys)
                        logger.log_figure(f"Final_Eval/Mesh_Sample_{i+1}", fig, epoch=num_epochs)
                        print(f"Generated plot {i+1}/{num_plots}")
                        
                    except ValueError as e:
                        print(f"Skipping visualization for mesh {mesh_idx}: {e}")

        total_wall_time = time.time() - global_start_time
        logger.close(total_wall_time)
        print("Pipeline execution finished!")

    cleanup_distributed(distributed)

if __name__ == "__main__":
    main()