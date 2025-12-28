from typing import Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import math
from sympy import true
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))

# Import all functions from utils_dir (handled by __init__.py)
# from utils_dir import 
from utils_dir.dataset import AlbumDataset
from utils_dir.dataset import spher_to_cart
from utils_dir import models

class ModelEval:
    """
    Evaluates a PyTorch model on a dataset and stores predictions, targets, and losses.
    
    Automatically runs evaluation on initialization and provides functionality to save
    results with timestamped filenames for experiment tracking.
    """

    class Image:
        def __init__(self, image_tensor: torch.Tensor):
            self.image_tensor = image_tensor

        def get_max_V(self):
            return self.image_tensor.max().item()

        def plot_image(self, save_path: Optional[Path] =None):
                    # Create figure
                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    
                    # Reshape logic
                    image2d = self.image_tensor.permute(2, 0, 1).reshape(-1, 1024)
                    
                    # Plot with 'viridis' (high contrast)
                    im = ax.imshow(image2d, aspect='auto', interpolation='none', cmap='viridis')
                    
                    # --- Visual Improvement: Channel Separators ---
                    # Draws a horizontal line every 24 rows to separate the stacked channels
                    for y_line in range(24, image2d.shape[0], 24):
                        ax.axhline(y_line - 0.5, color='white', linestyle='-', linewidth=1.5, alpha=0.8)

                    fig.colorbar(im, ax=ax, label='Voltage (V)', fraction=0.01, pad=0.01)

                    # Formatting
                    ax.set_ylabel("Index (Stacked 24 Channels x 4 Stations)")
                    ax.set_xlabel("Time Bin")
                    ax.set_title(f"Event Trace | Max V: {self.get_max_V():.4e}")
                    
                    # Optional: Add simple ticks for the blocks
                    ax.set_yticks(np.arange(12, 96, 24))
                    ax.set_yticklabels([f"Station {i}" for i in range(4)])

                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(save_path, bbox_inches='tight', dpi=100)
                        plt.close(fig) # Close to save memory if saving
                        print(f'Saving image to: {save_path}')
                    else:
                        plt.show()

    @dataclass
    class DataTensors:
        """
        Container for model evaluation results.
        
        Attributes:
            guess_arr: Tensor of model predictions [x, y, z] for each sample.
            target_arr: Tensor of ground truth labels [x, y, z] for each sample.
            loss_arr: Tensor of loss values for each sample.
            image_arr: List of voltage values
        """
        guess_arr: torch.Tensor
        target_arr: torch.Tensor
        loss_arr: torch.Tensor
        image_arr: List['ModelEval.Image']

    def __init__(self, 
                 model: torch.nn.Module,
                 data_set: AlbumDataset,
                 data_loader: torch.utils.data.DataLoader,
                 checkpoint_path: Optional[Path] = None,
                 loss_fn: torch.nn.Module = torch.nn.MSELoss(),
                 num_samples: int = 1000,
                 denormalize: bool = True,
                 store_images: bool = False):
        """
        Initialize evaluator and run evaluation immediately.
        
        Args:
            model: PyTorch model to evaluate.
            data_set: Dataset to evaluate on (must implement denormalize_label if denormalize=True).
            data_loader: Dataloader to evaluate on.
            checkpoint_path: Path to model checkpoint file or directory.
            loss_fn: Loss function to compute per-sample losses.
            num_samples: Number of samples to evaluate.
            denormalize: Choose to denormalize labels and guesses.
        """
        self.model = model
        self.data_set = data_set
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.num_samples = num_samples
        self.denormalize = denormalize
        self.store_images = store_images

        # --- Checkpoint Handling ---
        # Initialize default values
        self.checkpoint_dir = Path('')
        self.checkpoint_path = Path('')

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path) # Ensure it's a Path object
            if checkpoint_path.is_dir():
                print(f'Checkpoint path provided is a directory: {checkpoint_path}, will load latest checkpoint.')
                self.checkpoint_dir = checkpoint_path

                files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth') or f.endswith('.pt')]
                latest_checkpoint = files[-1]
                checkpoint_path = self.checkpoint_dir / latest_checkpoint
                self.checkpoint_path = checkpoint_path
            else:
                self.checkpoint_path = checkpoint_path
                self.checkpoint_dir = checkpoint_path.parent
                
            print(f'Loading checkpoint: {self.checkpoint_path.name}...')
            # Load checkpoint onto CPU to avoid CUDA OOM during simple eval setup
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Handle potential state_dict nesting
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(self._uncompile_keys(state_dict))
            print('Checkpoint Loaded!')
        else:
            print('No checkpoint specified, using current model state (untrained or pre-loaded).')

        # --- Run Evaluation ---
        self.data_tensors = self.get_xyz_preds_labels(inference=True)


    def _uncompile_keys(self, state_dict: dict) -> dict:
        """Helper to remove torch.compile prefix '_orig_mod.' from compiled state dict keys."""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "") 
            new_state_dict[new_key] = v
        return new_state_dict


    def get_xyz_preds_labels(self, inference: bool = True, verbose: bool = True) -> 'ModelEval.DataTensors':
        """
        Run model inference on dataset and collect predictions, labels, and losses.
        
        Args:
            inference: If True, runs in torch.inference_mode() for efficiency.
            verbose: Whether to print status updates.
        """
        guess_list = []
        target_list = []
        loss_list = []
        image_list = []

        if self.denormalize and verbose:
            print('Guesses and targets will be denormalized during processing.')

        # Select context manager
        context = torch.inference_mode() if inference else nullcontext()
        
        # Ensure model is in correct mode
        self.model.eval() if inference else self.model.train()

        with context:
            for idx, (image, label) in enumerate(self.data_loader):
                # Move data to same device as model
                device = next(self.model.parameters()).device
                image = image.to(device)
                label = label.to(device)

                # Forward pass
                pred = self.model(image)
                
                # Calculate loss (scalar)
                loss = self.loss_fn(pred.squeeze(), label.squeeze()).item()
                
                # Move to CPU for processing
                guess = pred.detach().cpu().squeeze()
                target = label.detach().cpu().squeeze()

                # Denormalize
                if self.denormalize:
                    guess = self.data_set.denormalize_label(guess)
                    target = self.data_set.denormalize_label(target)

                guess_list.append(guess)
                target_list.append(target)
                loss_list.append(loss)

                if self.store_images:
                    image_list.append(self.Image(image.squeeze()))

                if verbose:
                    print(f'\rTesting Model... ({idx+1}/{self.num_samples} samples)', end='', flush=True)

                if idx + 1 >= self.num_samples:
                    break
        
        if verbose:
            print() # Newline after progress bar

        # Stack lists into Tensors
        results = self.DataTensors(
            guess_arr=torch.stack(guess_list), 
            target_arr=torch.stack(target_list), 
            loss_arr=torch.tensor(loss_list),
            image_arr=image_list
        )

        return results


    def plot_true_reco(self, custom_title: Optional[str] = None, spherical: bool = False, show_maxV=False):
        """Plots x, y, z reconstruction comparison."""
        
        # Extract data (moving to numpy for plotting)
        guess_arr = self.data_tensors.guess_arr.numpy()
        target_arr = self.data_tensors.target_arr.numpy()
        maxV_arr = []
        if show_maxV:
            if self.data_tensors.image_arr is not None:
                maxV_arr = np.array([img.get_max_V() for img in self.data_tensors.image_arr])

        # Assuming format is [N, 3] corresponding to [x, y, z]
        x_true, y_true, z_true = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]
        x_guess, y_guess, z_guess = guess_arr[:, 0], guess_arr[:, 1], guess_arr[:, 2]

        fig, axes = plt.subplots(2, 3, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Helper to standardize plotting style
        def plot_component(fig, ax_col, true, guess, name, color, add_label=False, show_maxV=False):
            """
            ax_col: A column of axes (index 0 is main plot, index 1 is residual)
            add_label: Bool, only add the label for the first plot to avoid legend dupes
            show_maxV: If True, will show max voltage of of the image used to reconstruct.
            """
            
            # --- Top Plot (Scatter) ---
            sc1 = ax_col[0].scatter(true, guess, c=color, alpha=0.6, edgecolors='none')

            # Add Colorbar if needed
            if show_maxV and maxV_arr is not None:
                if add_label:
                    cbar = fig.colorbar(sc1)
                    cbar.set_label('Max V')
            
            # Determine label (Only label if add_label is True)
            lbl = 'Perfect Reconstruction' if add_label else None
            
            # Perfect reconstruction line
            low, high = min(true.min(), guess.min()), max(true.max(), guess.max())
            ax_col[0].plot([low, high], [low, high], color='tab:gray', ls='--', lw=3, label=lbl)
            
            ax_col[0].set_xlabel(f'{name} True')
            ax_col[0].set_ylabel(f'{name} Reco')
            ax_col[0].set_title(f'{name} Reconstructed vs. True')
            ax_col[0].grid(True, alpha=0.3)
            ax_col[0].set_aspect('equal', adjustable='datalim')

            # --- Bottom Plot (Residuals) ---
            ax_col[1].scatter(true, guess-true, c=color, alpha=0.6, edgecolors='none')
            
            # FIX: Use axhline instead of plotting np.zeros vs true
            # This draws one infinite line at y=0. Much faster and cleaner.
            ax_col[1].axhline(0, color='tab:gray', ls='--', lw=3)
            
            ax_col[1].grid(True, alpha=0.3)
            ax_col[1].set_ylabel(f'{name} Res.')


        # Setup colors for ease of control
        if show_maxV:
            orange = maxV_arr
            green = maxV_arr
            blue = maxV_arr
        else:
            orange = 'tab:orange'
            green = 'tab:green'
            blue = 'tab:blue'

        if not spherical:

            # Plot x (Add label HERE only)
            plot_component(fig, axes.T[0], x_true, x_guess, 'X', orange, add_label=True, show_maxV=show_maxV)
            
            # Plot y (No label)
            plot_component(fig, axes.T[1], y_true, y_guess, 'Y', green, add_label=False, show_maxV=show_maxV)
            
            # Plot z (No label)
            plot_component(fig, axes.T[2], z_true, z_guess, 'Z', blue, add_label=False, show_maxV=show_maxV)

            # Create legend only on the first plot (top left)
            # axes[0, 0] selects the top-left subplot
            axes[0, 0].legend(loc='upper left')

            title = custom_title if custom_title else f'Model performance for {self.checkpoint_path.name} (N={self.num_samples})'
            fig.suptitle(title, fontsize=20)
            fig.tight_layout()
            plt.show()

        if spherical:

            def cart_2_spher(x,y,z):
                r = np.sqrt(x**2+y**2+z**2)
                θ  = np.arccos(z / r) if np.any(r != 0) else 0
                φ  = np.arctan2(y, x)

                return r,θ,φ
                
            r_true, θ_true, φ_true = cart_2_spher(x_true,y_true,z_true)
            r_guess, θ_guess, φ_guess = cart_2_spher(x_guess,y_guess,z_guess)
            
            # Plot r (Add label HERE only)
            plot_component(fig, axes.T[0], r_true, r_guess, 'R', orange, add_label=true, show_maxV=show_maxV)
            
            # Plot θ (No label)
            plot_component(fig, axes.T[1], θ_true, θ_guess, r'$\theta$', green, add_label=False, show_maxV=show_maxV)
            
            # Plot φ (No label)
            plot_component(fig, axes.T[2], φ_true, φ_guess, r'$\phi$', blue, add_label=False, show_maxV=show_maxV)

            # Create legend only on the first plot (top left)
            # axes[0, 0] selects the top-left subplot
            axes[0, 0].legend(loc='upper left')

            title = custom_title if custom_title else f'Model performance for {self.checkpoint_path.name}'
            fig.suptitle(title, fontsize=20)
            fig.tight_layout()
            plt.show()

    def plot_vertex_analysis(self, custom_title: Optional[str] = None, spherical: bool = True):
            """Plots x, y, z reconstruction comparison with relative errors."""
            import matplotlib.pyplot as plt
            import numpy as np

            # Extract data
            guess_arr = self.data_tensors.guess_arr.numpy()
            target_arr = self.data_tensors.target_arr.numpy()

            # Assuming format is [N, 3] corresponding to [x, y, z]
            x_true, y_true, z_true = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]
            x_guess, y_guess, z_guess = guess_arr[:, 0], guess_arr[:, 1], guess_arr[:, 2]

            euclidean_dist_err = np.sqrt( (x_true-x_guess)**2 + (y_true-y_guess)**2 + (z_true-z_guess)**2 )
            euclidean_dist_err_68th = float(np.percentile(euclidean_dist_err,68.27))
            euclidean_dist_err_med = float(np.median(euclidean_dist_err))

            # --- Coordinate Transformation & Error Calc ---
            if spherical:
                def cart_2_spher(x, y, z):
                    r = np.sqrt(x**2 + y**2 + z**2)
                    # Clip z/r to [-1, 1] to avoid NaNs from floating point errors
                    theta = np.arccos(np.clip(z / (r + 1e-8), -1.0, 1.0))
                    phi = np.arctan2(y, x)
                    return r, theta, phi
                    
                r_true, theta_true, phi_true = cart_2_spher(x_true, y_true, z_true)
                r_guess, theta_guess, phi_guess = cart_2_spher(x_guess, y_guess, z_guess)

                # Calculate Errors
                x1_err, x2_err, x3_err = r_guess - r_true, theta_guess - theta_true, phi_guess - phi_true
                
                # Avoid division by zero
                x1_rel_err = x1_err / (r_true + 1e-8)
                x2_rel_err = x2_err / (theta_true + 1e-8)
                x3_rel_err = x3_err / (phi_true + 1e-8)

                names = ['Radius', 'Inclination', 'Azimuth']
                colors = ['tab:orange', 'tab:green', 'tab:blue']
                units = ['m', 'rad', 'rad']
                
            else:
                x1_err, x2_err, x3_err = x_guess - x_true, y_guess - y_true, z_guess - z_true
                
                x1_rel_err = x1_err / (x_true + 1e-8)
                x2_rel_err = x2_err / (y_true + 1e-8)
                x3_rel_err = x3_err / (z_true + 1e-8)

                names = ['X', 'Y', 'Z']
                colors = ['tab:orange', 'tab:green', 'tab:blue']
                units = ['m']*3

            # Obtain median of errors
            x1_err_med = float(np.median(x1_err))
            x2_err_med = float(np.median(x2_err))
            x3_err_med = float(np.median(x3_err))

            # Obtain median of relative errors
            x1_err_rel_med = float(np.median(x1_rel_err))
            x2_err_rel_med = float(np.median(x2_rel_err))
            x3_err_rel_med = float(np.median(x3_rel_err))

            # Obtain 68th percentiles of errors
            x1_err_68th = np.percentile(x1_err, [15.87, 84.13])
            x2_err_68th = np.percentile(x2_err, [15.87, 84.13])
            x3_err_68th = np.percentile(x3_err, [15.87, 84.13])

            # Obtain 68th percentiles of relative errors
            x1_rel_err_68th = np.percentile(x1_rel_err, [15.87, 84.13])
            x2_rel_err_68th = np.percentile(x2_rel_err, [15.87, 84.13])
            x3_rel_err_68th = np.percentile(x3_rel_err, [15.87, 84.13])

            # --- PLOTTING LOGIC (GridSpec) ---
            fig = plt.figure(figsize=(24, 10))
            # 2 Rows, 4 Columns. 
            # width_ratios: Make the first column (Euclidean) slightly wider (1.5x)
            gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1], hspace=0.3, wspace=0.3)

            # 1. Euclidean Plot (Spans BOTH rows in Column 0)
            ax_euc = fig.add_subplot(gs[:, 0])
            ax_euc.hist(euclidean_dist_err, bins=50, color='tab:gray', alpha=0.7, edgecolor='black')
            ax_euc.axvline(euclidean_dist_err_68th,color='black',label=f'68th Percentile = {euclidean_dist_err_68th:.3f} m',lw=2)
            ax_euc.axvline(euclidean_dist_err_med,color='black',label=f'Median: {euclidean_dist_err_med:.3f} m')
            ax_euc.set_title('Euclidean Distance Error')
            ax_euc.set_xlabel('Error (m)')
            ax_euc.set_ylabel('Frequency')
            ax_euc.legend()
            ax_euc.grid(True, alpha=0.3)

            # Helper to plot component columns
            def plot_col(col_idx, error_abs, error_abs68, error_abs_med, error_rel, error_rel68, error_rel_med, name, color, unit):
                # Top Row: Absolute Error
                ax_abs = fig.add_subplot(gs[0, col_idx])
                ax_abs.hist(error_abs, bins=50, color=color, alpha=0.7)
                ax_abs.axvline(error_abs68[0], color='black', label='68th Percentile', lw=2)
                ax_abs.axvline(error_abs68[1], color='black',lw=2)
                ax_abs.set_title(f'{name} Error\n Median: {error_abs_med:.3f} {unit} | 68th%: ({error_abs68[0]:.3f},{error_abs68[1]:.3f}) {unit}')
                ax_abs.set_xlabel('Abs Error')
                ax_abs.legend()
                ax_abs.grid(True, alpha=0.3)

                # Bottom Row: Relative Error
                ax_rel = fig.add_subplot(gs[1, col_idx])
                ax_rel.hist(error_rel, bins=50, color=color, alpha=0.5, hatch='//')
                ax_rel.axvline(error_rel68[0], color='black', label='68th Percentile', lw=2)
                ax_rel.axvline(error_rel68[1], color='black', lw=2)
                ax_rel.set_title(f'{name} Rel. Error\n Median: {error_rel_med:.4f} | 68th%: ({error_rel68[0]:.4f},{error_rel68[1]:.4f})')
                ax_rel.set_xlabel('Relative Error')
                ax_rel.legend()
                ax_rel.grid(True, alpha=0.3)

            # Plot the 3 components in Columns 1, 2, 3
            plot_col(1, x1_err, x1_err_68th, x1_err_med, x1_rel_err, x1_rel_err_68th, x1_err_rel_med, names[0], colors[0], units[0])
            plot_col(2, x2_err, x2_err_68th, x2_err_med, x2_rel_err, x2_rel_err_68th, x2_err_rel_med, names[1], colors[1], units[1])
            plot_col(3, x3_err, x3_err_68th, x3_err_med, x3_rel_err, x3_rel_err_68th, x3_err_rel_med, names[2], colors[2], units[2])

            # Final Formatting
            title = custom_title if custom_title else f'Model performance for {self.checkpoint_path.name} (N={self.num_samples})'
            fig.suptitle(title, fontsize=24, y=0.98)
            
            plt.show()

    def plot_progression_true_reco(self, spherical: bool = False, show_maxV: bool = False):
        """Iterates through all checkpoints in the checkpoint_dir, loads them,
        re-evaluates, and plots performance._summary_

        Args:
            spherical (bool, optional): If True, will plot spherical coordinates true vs reco. Defaults to False.
            show_maxV (bool, optional): If True, will show maxV of the image tested on plot. Defaults to False.

        Raises:
            ValueError: Raise error if no checkpoint dir specified to loop through.
            ValueError: Raise error if show_maxV is called without having stored images.
        """
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            raise ValueError('No checkpoint directory specified or directory does not exist.')

        if not self.store_images:
            raise ValueError('Cannot show max V if images are not stored. Use store_images=True when calling ModelEval')

        # Get list of files
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth') or f.endswith('.pt')]

        print(f"Found {len(files)} checkpoints to evaluate.")

        for checkpoint_file in files:
            full_path = self.checkpoint_dir / checkpoint_file
            
            print(f"\n--- Evaluating Checkpoint: {checkpoint_file} ---")
            
            # Load and Apply State Dict
            checkpoint = torch.load(full_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(self._uncompile_keys(state_dict))
            
            # Re-run evaluation
            self.data_tensors = self.get_xyz_preds_labels(verbose=True)
            
            # Plot
            self.plot_true_reco(custom_title=f'Progression: {checkpoint_file}', spherical= spherical,show_maxV = show_maxV)


    def save_images(self):
        """
        Save images for closer inspection
        """
        
        # Use the current working directory or script location
        save_dir = Path('model_results')
        save_dir.mkdir(parents=True, exist_ok=True)

        


def normalized_polar2cart(coord,dataset):
    r, theta, phi = coord
    r = r * dataset.get_rnorm()
    theta = theta*2*np.pi
    phi = phi*2*np.pi
    
    return [
         r * math.sin(phi) * math.cos(theta),
         r * math.sin(phi) * math.sin(theta),
         r * math.cos(phi)
    ]


def obtain_experiment_path_by_idx(experiments_dir: str, idx: int):
    experiments = [f for f in os.listdir(experiments_dir) if not f.startswith('.')]
    print(f'Experiment at id {idx}: {experiments[idx]}')
    return os.path.join(experiments_dir,experiments[idx])


def load_model_from_checkpoint(input_shape: int, hidden_units: int, output_shape: int, model_name, checkpoint = None, ):
    
    if checkpoint:
        checkpoint_state_dict = torch.load(checkpoint)['model_state_dict']
        
        model = model_name(input_shape=input_shape,
                                  hidden_units=hidden_units, 
                                  output_shape=output_shape)

        model.load_state_dict(torch.load(checkpoint_state_dict))
    else:
        model = model_name(input_shape=input_shape,
                                  hidden_units=hidden_units, 
                                  output_shape=output_shape)
        return model


def obtain_random_average_prediction_distance(album_data_loader,limit=None):
    if limit is None:
        limit = len(album_data_loader)
    
    x_distance_arr = []
    y_distance_arr = []
    z_distance_arr = []
    for batch,(X,y) in enumerate(album_data_loader):
        low = np.array([-1000, -1000, -3000])
        high = np.array([1000, 1000, 0])
        rand_pred = np.random.uniform(low,high)
        y = y.squeeze()
        x1, y1, z1 = rand_pred
        x2, y2, z2 = y
        x_distance = np.abs(x1-x2)
        x_distance_arr.append(x_distance)
        y_distance = np.abs(y1-y2)
        y_distance_arr.append(y_distance)
        z_distance = np.abs(z1-z2)
        z_distance_arr.append(z_distance)

        if batch == limit:
            break
    return x_distance_arr,y_distance_arr,z_distance_arr


def obtain_average_prediction_distance_radial(model,album_data_loader,limit=None):
    if limit is None:
        limit = len(album_data_loader)
    
    x_distance_arr = []
    y_distance_arr = []
    z_distance_arr = []
    with torch.inference_mode():
        for batch, (X,y) in enumerate(album_data_loader):
            pred = model(X)
            pred = pred.squeeze()
            y = y.squeeze()
            x1, y1, z1 = pred
            x2, y2, z2 = y
            x_distance = np.abs(x1-x2)
            x_distance_arr.append(x_distance)
            y_distance = np.abs(y1-y2)
            y_distance_arr.append(y_distance)
            z_distance = np.abs(z1-z2)
            z_distance_arr.append(z_distance)
            
            if batch == limit:
                break
    return x_distance_arr,y_distance_arr,z_distance_arr


def obtain_average_prediction_distance(model,album_data_loader,limit=None):
    if limit is None:
        limit = len(album_data_loader)
    
    x_distance_arr = []
    y_distance_arr = []
    z_distance_arr = []
    with torch.inference_mode():
        for batch, (X,y) in enumerate(album_data_loader):
            pred = model(X)
            pred = pred.squeeze()
            y = y.squeeze()
            x1, y1, z1 = pred
            x2, y2, z2 = y
            x_distance = np.abs(x1-x2)
            x_distance_arr.append(x_distance)
            y_distance = np.abs(y1-y2)
            y_distance_arr.append(y_distance)
            z_distance = np.abs(z1-z2)
            z_distance_arr.append(z_distance)
            
            if batch == limit:
                break
    return x_distance_arr,y_distance_arr,z_distance_arr


def get_vertex_prediction_distances_spher(model, album_data_set, album_data_loader, loss_fn = torch.nn.MSELoss(), max_event_num=0):
    class Distances:
        def __init__(self):
            self.dx = []
            self.dy = []
            self.dz = []
            self.dist = []
            self.loss = []

    num_events = len(album_data_loader)

    if max_event_num != 0:
        num_events = max_event_num

    with torch.inference_mode():
        distances = Distances()
        for batch, (X,y) in enumerate(album_data_loader):
            print(f'Looping... ({batch}/{num_events})')
            pred = model(X)
            y = y.squeeze()

            loss = loss_fn(pred, y).item()

            guess = pred.detach().numpy()
            target = y.squeeze().detach().numpy()

            x_targ,y_targ,z_targ = normalized_polar2cart(target,album_data_set)
            x_guess,y_guess,z_guess = normalized_polar2cart(guess,album_data_set)

            dx = x_guess - x_targ
            dy = y_guess - y_targ
            dz = z_guess - z_targ

            euclidean_dist = np.sqrt(dx**2 + dy**2 + dz**2)

            distances.dx.append(np.abs(dx))
            distances.dy.append(np.abs(dy))
            distances.dz.append(np.abs(dz))
            distances.dist.append(euclidean_dist)
            distances.loss.append(loss)

            if batch == num_events:
                break

        return distances


def compare_two_models(model1, model2, album_data_set, album_data_loader, comparison_type = 'spherical', loss_fn = torch.nn.MSELoss(), N_comparisons = 1000, untrained = False):
    distances_1 = get_vertex_prediction_distances_spher(model1, album_data_set,album_data_loader,loss_fn,N_comparisons)
    distances_2 = get_vertex_prediction_distances_spher(model2, album_data_set,album_data_loader,loss_fn,N_comparisons)
    return distances_1, distances_2


class Analytics:
    def __init__(self):
        self.euclidean_distance_error = []

        self.radius_guess = []
        self.radius_target = []
        self.radius_error = []
        self.rel_radius_error = []

        self.theta_guess = []
        self.theta_target = []
        self.theta_error = []
        self.rel_theta_error = []

        self.phi_guess = []
        self.phi_target = []
        self.phi_error = []
        self.rel_phi_error = []
        
        self.loss = []


def run_data_analysis(model, album_data_set, album_data_loader, loss_fn = torch.nn.MSELoss(), max_event_num=0) -> Analytics:
    num_events = len(album_data_loader)

    if max_event_num != 0:
        num_events = min(max_event_num, num_events)

    with torch.inference_mode():
        analysis = Analytics()
        for batch, (X, y) in enumerate(album_data_loader):
            if batch >= num_events:
                break
            print(f'\rLooping through event: ({batch}/{num_events})', end='')
            pred = model(X)
            y = y.squeeze()

            loss = loss_fn(pred, y).item()

            # move to CPU and ensure arrays are squeezed
            guess = pred.detach().cpu().numpy().squeeze()
            target = y.detach().cpu().numpy().squeeze()

            guess_r, guess_theta, guess_phi = album_data_set.denormalize_label(guess)
            target_r, target_theta, target_phi = album_data_set.denormalize_label(target)

            # make numpy arrays for safe arithmetic
            guess_r = np.array(guess_r)
            guess_theta = np.array(guess_theta)
            guess_phi = np.array(guess_phi)

            target_r = np.array(target_r)
            target_theta = np.array(target_theta)
            target_phi = np.array(target_phi)

            # absolute and relative errors
            radius_error = np.abs(guess_r - target_r)
            rel_radius_error = radius_error / (np.abs(target_r))

            # mitigate discontinuity at the boundary for guesses/targets close to theta=0
            theta_error_1 = np.abs(guess_theta - target_theta)
            theta_error_2 = np.abs(np.abs(guess_theta-2*np.pi) - target_theta)
            theta_error = min(theta_error_1,theta_error_2)
            rel_theta_error = theta_error / (np.abs(target_theta))

            phi_error = np.abs(guess_phi - target_phi)
            rel_phi_error = phi_error / (np.abs(target_phi))

            # convert spherical (r, theta, phi) to cartesian to compute euclidean distance
            # assuming theta is the polar angle (0..pi) and phi is the azimuth (0..2pi)
            gx = guess_r * np.cos(guess_theta) * np.sin(guess_phi)
            gy = guess_r * np.sin(guess_theta) * np.sin(guess_phi)
            gz = guess_r * np.cos(guess_phi)

            tx = target_r * np.cos(target_theta) * np.sin(target_phi)
            ty = target_r * np.sin(target_theta) * np.sin(target_phi)
            tz = target_r * np.cos(target_phi)

            euclidean_dist = np.sqrt((gx - tx) ** 2 + (gy - ty) ** 2 + (gz - tz) ** 2)

            # store results
            analysis.euclidean_distance_error.append(euclidean_dist)

            analysis.radius_guess.append(guess_r)
            analysis.radius_target.append(target_r)
            analysis.radius_error.append(radius_error)
            analysis.rel_radius_error.append(rel_radius_error)

            analysis.theta_guess.append(guess_theta)
            analysis.theta_target.append(target_theta)
            analysis.theta_error.append(theta_error)
            analysis.rel_theta_error.append(rel_theta_error)

            analysis.phi_guess.append(guess_phi)
            analysis.phi_target.append(target_phi)
            analysis.phi_error.append(phi_error)
            analysis.rel_phi_error.append(rel_phi_error)

            analysis.loss.append(loss)

        return analysis


def plot_model_analysis(analytics,event_num,distance=True,radius=True,theta=True,phi=True,degrees=True):

    if degrees:
        unit = 'deg'
    else:
        unit = 'rad'

    if distance:
        data = analytics.euclidean_distance_error
        mean = np.mean(data)
        plt.figure(figsize=(12,6))
        plt.title(f'Euclidean Distance Error for {event_num} events')
        counts = plt.hist(data,zorder=0,bins=50)
        max_freq = np.max(counts[0])
        max_index = max(range(len(counts[0])), key=counts[0].__getitem__)
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f} m',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f} m',color='black',ls='--',lw=2,zorder=1)
        plt.xlabel('Euclidean Distance Error (m)')
        plt.ylabel('Frequency')
        plt.grid(True,alpha=0.5)
        plt.legend()
        plt.show()
    if radius:
        color = 'tab:orange'
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(1,2,1)
        data = analytics.radius_error
        mean = np.mean(data)
        plt.title(f'Radius Error for {event_num} events')
        counts = ax1.hist(data,zorder=0,bins=50,color=color)
        max_freq = np.max(counts[0])
        max_index = max(range(len(counts[0])), key=counts[0].__getitem__)
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f} m',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f} m',color='black',ls='--',lw=2,zorder=1)
        ax1.set_xlabel('Radius Distance Error (m)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True,alpha=0.5)
        ax1.legend()

        ax2 = plt.subplot(1,2,2)
        data = analytics.rel_radius_error
        mean = np.mean(data)
        plt.title(f'Relative Radius Error for {event_num} events')
        counts = ax2.hist(data,zorder=0,bins=50,color=color)
        max_freq = np.max(counts[0])
        max_index = max(range(len(counts[0])), key=counts[0].__getitem__)
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f}',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f}',color='black',ls='--',lw=2,zorder=1)
        ax2.set_xlabel('Relative Radius Distance Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True,alpha=0.5)
        ax2.legend()

    if theta:
        color = 'tab:green'
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(1,2,1)
        data = np.rad2deg(analytics.theta_error)
        mean = np.mean(data)
        plt.title(f'Phi Error for {event_num} events')
        counts = ax1.hist(data,zorder=0,bins=50,color=color)
        max_freq = np.max(counts[0])
        max_index = max(range(len(counts[0])), key=counts[0].__getitem__)
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f} {unit}',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f} {unit}',color='black',ls='--',lw=2,zorder=1)
        ax1.set_xlabel(f'Phi Distance Error ({unit})')
        ax1.set_ylabel('Frequency')
        ax1.grid(True,alpha=0.5)
        ax1.legend()

        ax2 = plt.subplot(1,2,2)
        data = analytics.rel_theta_error
        mean = np.mean(data)
        plt.title(f'Relative Phi Error for {event_num} events')
        counts = ax2.hist(data,zorder=0,bins=50,color=color)
        max_freq = np.max(counts[0])
        max_index = max(range(len(counts[0])), key=counts[0].__getitem__)
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f}',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f}',color='black',ls='--',lw=2,zorder=1)
        ax2.set_xlabel('Relative Phi Distance Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True,alpha=0.5)
        ax2.legend()

    if phi:
        color = 'tab:pink'
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(1,2,1)
        data = np.rad2deg(analytics.phi_error)
        mean = np.mean(data)
        plt.title(f'Theta Error for {event_num} events')
        counts = ax1.hist(data,zorder=0,bins=50,color=color)
        max_freq = np.max(counts[0])
        max_index = max(range(len(counts[0])), key=counts[0].__getitem__)
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f} {unit}',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f} {unit}',color='black',ls='--',lw=2,zorder=1)
        ax1.set_xlabel(f'Theta Distance Error ({unit})')
        ax1.set_ylabel('Frequency')
        ax1.grid(True,alpha=0.5)
        ax1.legend()

        ax2 = plt.subplot(1,2,2)
        data = analytics.rel_phi_error
        mean = np.mean(data)
        plt.title(f'Relative Theta Error for {event_num} events')
        counts = ax2.hist(data,zorder=0,bins=50,color=color)
        max_freq = np.max(counts[0])
        mode = (counts[1][max_index]+counts[1][max_index+1])/2
        plt.axvline(mean,0,max_freq,label=f'Mean: {mean:.4f}',color='black',lw=2,zorder=1)
        plt.axvline(mode,0,max_freq,label=f'Mode: {mode:.4f}',color='black',ls='--',lw=2,zorder=1)
        ax2.set_xlabel('Relative Theta Distance Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True,alpha=0.5)
        ax2.legend()
