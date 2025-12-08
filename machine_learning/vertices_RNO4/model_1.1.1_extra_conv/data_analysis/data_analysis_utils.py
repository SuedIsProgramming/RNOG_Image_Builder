from typing import Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from contextlib import nullcontext
from pathlib import Path
import datetime
import numpy as np
import math
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))

# Import all functions from utils_dir (handled by __init__.py)
from utils_dir import *


class ModelEval:
    """
    Evaluates a PyTorch model on a dataset and stores predictions, targets, and losses.
    
    Automatically runs evaluation on initialization and provides functionality to save
    results with timestamped filenames for experiment tracking.
    """

    @dataclass
    class DataTensors:
        """
        Container for model evaluation results.
        
        Attributes:
            guess_arr: Tensor of model predictions [r, theta, phi] for each sample.
            target_arr: Tensor of ground truth labels [r, theta, phi] for each sample.
            loss_arr: Tensor of loss values for each sample.
        """
        guess_arr: torch.Tensor
        target_arr: torch.Tensor
        loss_arr: torch.Tensor

    def __init__(self, 
                 model: torch.nn.Module,
                 data_set: torch.utils.data.Dataset,
                 data_loader: torch.utils.data.DataLoader,
                 checkpoint_path: Optional[Path] = None,
                 loss_fn: torch.nn.Module = torch.nn.MSELoss(),
                 num_samples: int = 1000,
                 denormalize: bool = True):
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

        # --- Checkpoint Handling ---
        # Initialize default values
        self.checkpoint_dir = Path('')
        self.checkpoint_path = Path('')

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path) # Ensure it's a Path object
            if checkpoint_path.is_dir():
                print(f'Checkpoint path provided is a directory: {checkpoint_path}')
                self.checkpoint_dir = checkpoint_path
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
        self.data_tensors = self.get_rθφ_preds_labels(inference=True)

    def _uncompile_keys(self, state_dict: dict) -> dict:
        """Helper to remove torch.compile prefix '_orig_mod.' from compiled state dict keys."""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "") 
            new_state_dict[new_key] = v
        return new_state_dict

    def get_rθφ_preds_labels(self, inference: bool = True, verbose: bool = True) -> 'ModelEval.DataTensors':
        """
        Run model inference on dataset and collect predictions, labels, and losses.
        
        Args:
            inference: If True, runs in torch.inference_mode() for efficiency.
            verbose: Whether to print status updates.
        """
        guess_list = []
        target_list = []
        loss_list = []

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
                guess_np = pred.detach().cpu().squeeze()
                target_np = label.detach().cpu().squeeze()

                # Denormalize (Requires dataset to have 'denormalize_label' method)
                if self.denormalize:
                    guess_np = self.data_set.denormalize_label(guess_np.numpy())
                    target_np = self.data_set.denormalize_label(target_np.numpy())

                guess_list.append(guess_np)
                target_list.append(target_np)
                loss_list.append(loss)

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
            loss_arr=torch.tensor(loss_list)
        )
        return results

    def plot_true_reco(self, custom_title: str = None):
            """Plots R, Theta, Phi reconstruction comparison."""
            
            # Extract data (moving to numpy for plotting)
            guess_arr = self.data_tensors.guess_arr.numpy()
            target_arr = self.data_tensors.target_arr.numpy()

            # Assuming format is [N, 3] corresponding to [r, theta, phi]
            r_true, θ_true, φ_true = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]
            r_guess, θ_guess, φ_guess = guess_arr[:, 0], guess_arr[:, 1], guess_arr[:, 2]

            fig, axes = plt.subplots(2, 3, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Helper to standardize plotting style
            def plot_component(ax_col, true, guess, name, color, add_label=False):
                """
                ax_col: A column of axes (index 0 is main plot, index 1 is residual)
                add_label: Bool, only add the label for the first plot to avoid legend dupes
                """
                
                # --- Top Plot (Scatter) ---
                ax_col[0].scatter(true, guess, color=color, alpha=0.6, edgecolors='none')
                
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
                ax_col[1].scatter(true, guess-true, color=color, alpha=0.6, edgecolors='none')
                
                # FIX: Use axhline instead of plotting np.zeros vs true
                # This draws one infinite line at y=0. Much faster and cleaner.
                ax_col[1].axhline(0, color='tab:gray', ls='--', lw=3)
                
                ax_col[1].grid(True, alpha=0.3)
                ax_col[1].set_ylabel(f'{name} Res.')

            # Plot R (Add label HERE only)
            plot_component(axes.T[0], r_true, r_guess, 'Radius', 'tab:orange', add_label=True)
            
            # Plot Theta (No label)
            plot_component(axes.T[1], θ_true, θ_guess, r'$\theta$', 'tab:green', add_label=False)
            
            # Plot Phi (No label)
            plot_component(axes.T[2], φ_true, φ_guess, r'$\phi$', 'tab:blue', add_label=False)
            
            # Create legend only on the first plot (top left)
            # axes[0, 0] selects the top-left subplot
            axes[0, 0].legend(loc='upper left')

            title = custom_title if custom_title else f'Model performance for {self.checkpoint_path.name}'
            fig.suptitle(title, fontsize=20)
            fig.tight_layout()
            plt.show()


    def plot_progression_true_reco(self):
        """ 
        Iterates through all checkpoints in the checkpoint_dir, loads them,
        re-evaluates, and plots performance.
        """
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            raise ValueError('No checkpoint directory specified or directory does not exist.')

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
            self.data_tensors = self.get_rθφ_preds_labels(verbose=True)
            
            # Plot
            self.plot_true_reco(custom_title=f'Progression: {checkpoint_file}')

    def save_model_results(self):
        """
        Save evaluation results to disk with timestamped filename.
        """
        model_name = self.model.__class__.__name__
        
        # Use the current working directory or script location
        save_dir = Path('model_results')
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'{model_name}_{timestamp}.pt'
        save_path = save_dir / filename

        # FIX: self.model_results did not exist. Used self.data_tensors instead.
        payload = {'results': self.data_tensors}

        torch.save(payload, save_path)
        print(f"Results saved to {save_path}")

def polar2cart(coord):
    r, theta, phi = coord
    
    return [
         r * math.sin(phi) * math.cos(theta),
         r * math.sin(phi) * math.sin(theta),
         r * math.cos(phi)
    ]

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
