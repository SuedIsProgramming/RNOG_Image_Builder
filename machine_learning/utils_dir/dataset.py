from decimal import DivisionByZero
from torch.utils.data import Dataset
from h5py import File
import numpy as np
import threading
import torch
import os

def spher_to_cart(label):
    r,theta,phi = label

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    cartesian_label = label = torch.tensor([x, y, z], dtype=torch.float32)

    return cartesian_label

class AlbumDataset(Dataset):

    """
    Custom PyTorch Dataset for loading particle physics data from HDF5 files.
    
    This dataset handles large HDF5 files containing particle detector images and 
    corresponding vertex position labels. Uses thread-local storage to maintain
    separate file handles per worker thread for efficient multi-threaded loading.
    
    Structure expected in HDF5 file:
    - Root level: event groups (e.g., 'event1', 'event2', ...)
    - Each event group contains:
        - 'image': 2D detector image data
        - 'label': 3D vertex coordinates [x, y, z]
    """
    
    def __init__(self, album_path, transform=None, target_transform=None, preload_keys=True,normalize_labels=True,normalization_factors = None):
        """
        Initialize the AlbumDataset.
        
        Args:
            album_path (str): Path to the HDF5 file containing the dataset
            transform (callable, optional): Transform to apply to images
            target_transform (callable, optional): Transform to apply to labels
            preload_keys (bool): Whether to preload all event keys into memory. If True improves performance but uses more memory.
            normalize_labels (bool): Will calculate mean and std. of labels and normalize utilizing the z-number.
        """
        # Store configuration
        self.path = album_path
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_labels = normalize_labels
        
        # Validate file exists
        if not os.path.exists(album_path):
            raise FileNotFoundError(f"Album file not found: {album_path}")

        # Calculate and store file size for monitoring
        self.size = f'Size of file: {os.path.getsize(album_path)*1e-9:.4f} GB'

        # Preload all event keys for faster access (optional optimization)
        self.preload_keys = preload_keys
        with File(self.path, 'r') as file:
            if self.preload_keys:
                self.event_keys = list(file.keys())
            else:
                self.event_keys = None
            self.num_images = len(file.keys())

        # Save local data for thread.
        self._local = threading.local()

        # Validation - ensure we have data
        if self.num_images == 0:
            raise ValueError(f"No events found in {album_path}")
        
                # Compute normalization statistics if requested
        if self.normalize_labels:
            if normalization_factors is None:
                print("Computing normalization statistics...")
                self._compute_normalization_stats()
                print("Normalization stats computed:")
                print(f"  r:     mean={self.r_mean:.4f}, std={self.r_std:.4f}")
                print(f"  theta: mean={self.theta_mean:.4f}, std={self.theta_std:.4f}")
                print(f"  phi:   mean={self.phi_mean:.4f}, std={self.phi_std:.4f}")
                print(f'[{self.r_mean},{self.r_std},{self.theta_mean},{self.theta_std},{self.phi_mean},{self.phi_std}]')
            else:
                print("Utilizing inputted normalization statistics")
                self.r_mean, self.r_std, self.theta_mean, self.theta_std, \
                self.phi_mean, self.phi_std = normalization_factors
                print(f"  r:     mean={self.r_mean:.4f}, std={self.r_std:.4f}")
                print(f"  theta: mean={self.theta_mean:.4f}, std={self.theta_std:.4f}")
                print(f"  phi:   mean={self.phi_mean:.4f}, std={self.phi_std:.4f}")
                print(f'[{self.r_mean},{self.r_std},{self.theta_mean},{self.theta_std},{self.phi_mean},{self.phi_std}]')
        else:
            # Set to None to indicate no normalization
            self.r_mean = None
            self.r_std = None
            self.theta_mean = None
            self.theta_std = None
            self.phi_mean = None
            self.phi_std = None

    def _compute_normalization_stats(self):
        """
        Compute mean and std for r, theta, phi across entire dataset.
        This is called once during __init__ if normalize_labels=True.
        """
        r_values = []
        theta_values = []
        phi_values = []
        
        with File(self.path, 'r') as f:
            for idx in range(self.num_images):
                print(f'\rCompounding statistics... ({idx}/{self.num_images})',end='')
                _, label = self.__getitem__(idx,to_normalize=True)
                r, theta, phi = label
                
                r_values.append(r)
                theta_values.append(theta)
                phi_values.append(phi)
        
        # Convert to numpy arrays for efficient computation
        r_values = np.array(r_values)
        theta_values = np.array(theta_values)
        phi_values = np.array(phi_values)
        
        # Compute statistics
        self.r_mean = float(np.mean(r_values))
        self.r_std = float(np.std(r_values))
        
        self.theta_mean = float(np.mean(theta_values))
        self.theta_std = float(np.std(theta_values))
        
        self.phi_mean = float(np.mean(phi_values))
        self.phi_std = float(np.std(phi_values))
        
        # Avoid division by zero
        if self.r_std < 1e-8:
            raise DivisionByZero
            # print('r_std is zero! Double check data')
            # self.r_std = 1.0
        if self.theta_std < 1e-8:
            raise DivisionByZero
            # print('theta_std is zero! Double check data')
            # self.theta_std = 1.0
        if self.phi_std < 1e-8:
            raise DivisionByZero
            # print('phi_std is zero! Double check data')
            # self.phi_std = 1.0

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_images

    def __getitem__(self, idx,to_normalize=False):
        """
        Fetch a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, label) where:
                - image: torch.Tensor of shape (1, H, W) - detector image with channel dim
                - label: torch.Tensor of shape (3,) - vertex coordinates [x, y, z]
        """
        # Validate index bounds
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_images}")
        
        # Get or create thread-local file handle
        # This ensures each DataLoader worker has its own file handle
        if not hasattr(self._local, 'file_handle') or self._local.file_handle is None:
            try:
                self._local.file_handle = File(self.path, 'r')
            except Exception as e:
                raise RuntimeError(f"Failed to open HDF5 file: {e}")

        # Determine event key - use preloaded keys if available, otherwise construct
        if self.event_keys:
            event_key = self.event_keys[idx]
        else:
            event_key = f'event{idx+1}'  # Assumes 1-indexed event naming

        file_handle = self._local.file_handle

        try:
            # Load image data
            # Convert numpy array to PyTorch tensor with float32 dtype
            image = torch.from_numpy(np.array(file_handle[event_key]['image'])).float()
            
            # Load label data (vertex coordinates)
            label = torch.from_numpy(np.array(file_handle[event_key]['label'])).float()
            
        except KeyError as e:
            raise KeyError(f"Event key '{event_key}' not found in dataset or missing 'image'/'label': {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data for {event_key}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Add channel dimension (H, W) -> (1, H, W)
        # Most CNN architectures expect a channel dimension
        image = torch.unsqueeze(image, 0)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Apply normalization if enabled
        if self.normalize_labels and not to_normalize:
            r, theta, phi = label
            
            # Z-score normalization using precomputed statistics
            r_normalized = (r - self.r_mean) / self.r_std
            theta_normalized = (theta - self.theta_mean) / self.theta_std
            phi_normalized = (phi - self.phi_mean) / self.phi_std
            
            label = torch.tensor([r_normalized, theta_normalized, phi_normalized], dtype=torch.float32)

        return image, label

    def denormalize_label(self, normalized_label):
            """
            Convert normalized label back to original units.
            
            Args:
                normalized_label: torch.Tensor of shape (3,) or (batch, 3) with normalized [r, theta, phi]
            
            Returns:
                torch.Tensor with denormalized values
            """
            if not self.normalize_labels:
                return normalized_label
            
            if not isinstance(normalized_label, torch.Tensor):
                normalized_label = torch.from_numpy(normalized_label)

            if normalized_label.dim() == 1:
                r_norm, theta_norm, phi_norm = normalized_label
                r = r_norm * self.r_std + self.r_mean
                theta = theta_norm * self.theta_std + self.theta_mean
                phi = phi_norm * self.phi_std + self.phi_mean
                return torch.tensor([r, theta, phi], dtype=torch.float32)
            else:
                # Batch of labels
                r_norm = normalized_label[:, 0] * self.r_std + self.r_mean
                theta_norm = normalized_label[:, 1] * self.theta_std + self.theta_mean
                phi_norm = normalized_label[:, 2] * self.phi_std + self.phi_mean
                return torch.stack([r_norm, theta_norm, phi_norm], dim=1)

    def get_normalization_factors(self):
        normalization_factors = [self.r_mean, self.r_std, self.theta_mean, self.theta_std, \
                                 self.phi_mean, self.phi_std]
        return normalization_factors

    def close(self):
        """
        Close any open file handles.
        
        Should be called when dataset is no longer needed to free resources.
        """
        if hasattr(self._local, 'file_handle') and self._local.file_handle is not None:
            try:
                self._local.file_handle.close()
                self._local.file_handle = None
            except:
                pass  # Ignore errors during cleanup

    def __del__(self):
        """Clean up file handles"""
        self.close()