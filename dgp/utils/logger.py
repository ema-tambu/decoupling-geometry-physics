import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import csv
import statistics

# Try to import pynvml for GPU tracking. If missing, we gracefully degrade.
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not found. GPU hardware utilization will not be logged.")

class ExperimentLogger:
    """
    Handles TensorBoard, Text Logging, Figure Saving, and Hardware Monitoring.
    Strictly restricted to the main process (Rank 0) to prevent duplicate I/O.
    """
    def __init__(self, config, is_main_process=True):
        self.is_main_process = is_main_process
        if not self.is_main_process:
            return

        self.config = config
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create timestamped experiment directory (job id / pid suffix avoids collisions between concurrent runs)
        suffix = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        self.exp_dir = os.path.join("runs", f"exp_{self.timestamp}_{suffix}")
        self.plot_dir = os.path.join(self.exp_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(log_dir=self.exp_dir)
        
        # Initialize Text Log
        self.log_file_path = os.path.join(self.exp_dir, "training_log.txt")
        self._init_log_file()
        
        # Initialize GPU Monitoring
        self._init_nvml()

        self.history =[]
        self.epoch_times = []
        self.gpu_utils = []

    @staticmethod
    def _hardware():
        import socket
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return {
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "world_size": os.environ.get("WORLD_SIZE", "1"),
            "gpus_visible": n,
            "gpu_model": torch.cuda.get_device_name(0) if n else "cpu",
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count())),
            "torch": torch.__version__,
        }

    def _init_log_file(self):
        with open(self.log_file_path, "w") as f:
            f.write(f"Training Log - {self.timestamp}\n")
            f.write("="*50 + "\n")
            f.write("Hardware:\n")
            for k, v in self._hardware().items():
                f.write(f"  {k}: {v}\n")
            f.write("Configuration:\n")
            
            # Identify active architecture to filter out irrelevant hyperparameters
            active_arch = self.config.get('model', {}).get('architecture', 'cross_linear')
            
            # Explicitly define parameters that belong exclusively to specific models
            cross_linear_keys = ['N', 'C_linear', 'M_linear', 'L_linear']
            perceiver_keys = ['F_perceiver', 'E_perceiver', 'M_perceiver', 'num_mixer_layers', 'mixer_hidden_layers']
            
            for section, params in self.config.items():
                f.write(f"  [{section}]\n")
                
                # Check if params is a dictionary before iterating over items
                if isinstance(params, dict):
                    for k, v in params.items():
                        # --- Filtering Logic for Model Parameters ---
                        if section == 'model':
                            if active_arch not in ("cross_linear", "siren") and k in cross_linear_keys:
                                continue
                            if active_arch != "perceiver" and k in perceiver_keys:
                                continue
                            if active_arch != "siren" and k.endswith("_siren"):
                                continue
                            if active_arch != "gino" and k.endswith("_gino"):
                                continue
                        # --------------------------------------------
                                
                        f.write(f"    {k}: {v}\n")
                else:
                    f.write(f"    {params}\n")
                    
            f.write("="*50 + "\n\n")

    def _init_nvml(self):
        self.device_count = 0
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                print("Warning: NVML initialized but failed to get devices.")

    def log_model_params(self, num_params):
        if not self.is_main_process:
            return
        with open(self.log_file_path, "a") as f:
            f.write(f"Total Model Parameters: {num_params:,}\n")
            f.write("="*50 + "\n\n")

    def log_metrics(self, epoch, train_loss, val_loss, epoch_duration, custom_metrics=None, is_new_best=False):
        if not self.is_main_process:
            return
        
        self.epoch_times.append(epoch_duration)
        
        self.tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        self.tb_writer.add_scalar("Loss/Validation", val_loss, epoch)
        
        custom_metrics_str = ""
        row_data = {
            "Epoch": epoch, 
            "Train_Loss": train_loss, 
            "Val_Loss": val_loss, 
            "Time_s": round(epoch_duration, 4)
        }

        if custom_metrics:
            for name, value in custom_metrics.items():
                self.tb_writer.add_scalar(f"Metrics/{name}", value, epoch)
                custom_metrics_str += f" | {name}: {value:.4e}"
                row_data[name] = value

        avg_gpu = 0.0
        if NVML_AVAILABLE and self.device_count > 0:
            current_utils = []
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                current_utils.append(util.gpu)
                self.tb_writer.add_scalar(f"Hardware_GPU_{i}/Compute_Utilization_%", util.gpu, epoch)
                self.tb_writer.add_scalar(f"Hardware_GPU_{i}/Memory_Used_MB", mem.used / (1024**2), epoch)
            
            avg_gpu = sum(current_utils) / len(current_utils)
            self.gpu_utils.append(avg_gpu)
            row_data["GPU_Util_Pct"] = round(avg_gpu, 4)

        self.history.append(row_data)

        marker = "  *** NEW BEST ***" if is_new_best else ""
        log_line = f"Epoch {epoch:04d} [{epoch_duration:.4f}s] | Train: {train_loss:.6e} | Val: {val_loss:.6e}{custom_metrics_str}{marker}\n"
        
        print(log_line.strip())
        with open(self.log_file_path, "a") as f:
            f.write(log_line)

    def log_figure(self, tag, fig, epoch):
        """Saves a matplotlib figure to both TensorBoard and the timestamped plots folder."""
        if not self.is_main_process:
            return
        
        self.tb_writer.add_figure(tag, fig, epoch)
        
        # 1. Construct the full path
        save_path = os.path.join(self.plot_dir, f"{tag}_epoch_{epoch:04d}.png")
        
        # 2. Extract the directory part of the path and ensure it exists!
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def close(self, total_wall_time=None):
        if not self.is_main_process:
            return
            
        # 1. Write the CSV File
        csv_path = os.path.join(self.exp_dir, "metrics.csv")
        if self.history:
            keys = self.history[0].keys()
            with open(csv_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.history)

        # 2. Compute and Log Final Statistics
        if self.epoch_times:
            mean_time = statistics.mean(self.epoch_times)
            std_time = statistics.stdev(self.epoch_times) if len(self.epoch_times) > 1 else 0.0
            
            mean_gpu = statistics.mean(self.gpu_utils) if self.gpu_utils else 0.0
            std_gpu = statistics.stdev(self.gpu_utils) if len(self.gpu_utils) > 1 else 0.0

            with open(self.log_file_path, "a") as f:
                f.write("\n" + "="*50 + "\n")
                f.write("Training Performance Summary:\n")
                
                # log the TRUE total time from start to finish
                if total_wall_time is not None:
                    # let's render the total time in a hh:mm:ss format for better readability
                    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_wall_time))
                    f.write(f"  Total Wall Time: {total_time_str} (hh:mm:ss)\n")

                f.write(f"  Compute Time per Epoch: {mean_time:.4f} ± {std_time:.4f} seconds\n")
                
                if NVML_AVAILABLE:
                    f.write(f"  Average GPU Utilization: {mean_gpu:.4f}% ± {std_gpu:.4f}%\n")
                f.write("="*50 + "\n")

        self.tb_writer.close()
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass