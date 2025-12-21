from torch.utils.data import DataLoader
import logging
import torch
import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all functions from utils_dir (handled by __init__.py)
from utils_dir import *

# Only function if current working directory is model dir:

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

if 'model_' not in current_dir:
    raise ValueError("Must run from a directory containing 'model_' in the name")
    
if 'experiments' in current_dir:
    raise ValueError("Cannot run from within experiments directory")
    
print('✅ Inside correct folder')

# PARAMS =============================================================
# Setup Batch size
BATCH_SIZE = 128
# Setup number of epochs to train
NUM_EPOCHS = int(1e6)
# Use checkpoint if needed
# checkpoint_name = 'checkpoint_e500.pth'
# Learning Rate
LEARNING_RATE = 0.01
# ====================================================================

# Print out versions and device to make sure everything is working
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)
print(device)

# Paths to album:
# album_path = '/data/i3store/users/ssued/albums/RNO_benchmark.hdf5'
train_album_path = '/data/i3store/users/ssued/albums/RNO_album_10_13_2025_10k/album_RNO4.hdf5_train.hdf5'
test_album_path = '/data/i3store/users/ssued/albums/RNO_album_10_13_2025_10k/album_RNO4.hdf5_test.hdf5'

# Load Datasets
# album = AlbumDataset(album_path, transform=None, target_transform=None)
train_album = AlbumDataset(train_album_path, transform=None, target_transform=None, label_transform='spherical')
test_album = AlbumDataset(test_album_path, transform=None, target_transform=None, label_transform='spherical')

print(f'Train album size: {train_album.num_images} | Test album size: {test_album.num_images}')

# Load DataLoaders
train_data_loader = DataLoader(dataset = train_album,
                               batch_size = BATCH_SIZE,
                               shuffle = True,
                               num_workers = 8)
test_data_loader = DataLoader(dataset = test_album,
                              batch_size = BATCH_SIZE,
                              shuffle = False,
                              num_workers = 4)

print(f'Number of train batches: {len(train_data_loader)} | Number of test batches: {len(test_data_loader)}')

# Initialize model
models = RNO_four_1_0_0_batch_norm(input_shape=1,
                          hidden_units=20, 
                          output_shape=3,
                          num_epochs=NUM_EPOCHS,
                          batch_size=BATCH_SIZE,
                          num_train_batches=len(train_data_loader)
                         )

# Setup optimizer
optimizer = torch.optim.Adam(params=models.parameters(), lr = LEARNING_RATE)
optimizer_name = optimizer.__class__.__name__
# Setup loss function
#loss_fn = torch.nn.HuberLoss(delta=50)
loss_fn = torch.nn.MSELoss()
loss_fn_name = loss_fn.__class__.__name__

experiment_name = (f'exp_e{NUM_EPOCHS}' +
                  f'_bn{BATCH_SIZE}' +
                  f'_tr{len(train_data_loader)}' +
                  f'_te{len(test_data_loader)}' +
                  f'_lfn-{loss_fn_name}' +
                  f'_opt-{optimizer_name}' +
                  f'_lr-{LEARNING_RATE}' +
                  f'_tanh-no_batchnrm-no_lReLU-yes')



# Create experiments directory if it doesn't exist
os.makedirs('experiments', exist_ok=True)

# Create specific experiment directory if it doesn't exist
idx = 1
experiment_path = os.path.join('experiments', experiment_name)
while os.path.exists(experiment_path):
    experiment_path = f"{experiment_path}_{idx}"
    idx += 1
os.makedirs(experiment_path, exist_ok=True)

# Setup logging
logger = logging.getLogger('experiment_log') # Setup logging
logging.basicConfig(filename=f'{experiment_path}/experiment.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='[%(levelname)s: %(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger.info(f"Starting experiment: {experiment_name}")
logger.info(f"Device: {device}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Model: {type(models).__name__}")
logger.info(f"Optimizer: {optimizer_name}")
logger.info(f"Loss function: {loss_fn_name}")

train_test(model = models, 
           train_dataloader = train_data_loader, 
           test_dataloader = test_data_loader, 
           optimizer = optimizer,
           scheduler = None,
           loss_fn = loss_fn,
           device = device,
           experiment_name = experiment_name,
           epochs = NUM_EPOCHS,
           checkpoint_freq = 100,
           checkpoint_name = None,
           loss_file = 'losses.txt',
           logger = logger)