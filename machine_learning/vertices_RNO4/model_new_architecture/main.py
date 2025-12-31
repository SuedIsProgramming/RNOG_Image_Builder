from torch.utils.data import DataLoader
import logging
import torch
import time
import sys
import os

# Enable TF32 for faster matrix multiplications
torch.set_float32_matmul_precision('high')

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all functions from utils_dir (handled by __init__.py)
from utils_dir import train_test, AlbumDataset, models, spher_to_cart

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
NUM_EPOCHS = int(100_000)
# Use checkpoint if needed (None if not)
checkpoint_path = None
# Checkpoint frequency for saving
CHECKPOINT_FREQ = 50
# Choose whether to include initial batch normalization
NORMALIZE_INPUTS = True
# How steep the slope for the negative domain of LReLU is, 0 = ReLU.
LEAK_FACTOR = 0.1 # Usually 0.1
#Dropout Rate
DROPOUT_RATE = 0.1 # Usually 0.1
# Learning Rate
LEARNING_RATE = 0.001
# Hidden Units
HIDDEN_UNITS = 32
# Transform
CARTESIAN = True
# Wandb ID
WANDB_ID = None
# Temporal resolution of second layer
TEMPORAL_RES = 32
# ====================================================================

if checkpoint_path is not None:
    print('<<<<<<<<<<WARNING: UTILIZING CHECKPOINT>>>>>>>>>>')
if WANDB_ID is not None:
    print(f'<<<<<<<<<<WARNING: UTILIZING RUN_ID: {WANDB_ID}>>>>>>>>>>')

if CARTESIAN:
    TARGET_TRANSFORM = spher_to_cart
else:
    TARGET_TRANSFORM = None

# Print out versions and device to make sure everything is working
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = "cpu"
    
print(torch.__version__)
print(device)

# Paths to album:
# album_path = '/data/i3store/users/ssued/albums/RNO_benchmark.hdf5'
train_album_path = '/data/i3store/users/ssued/albums/RNO_album_11_03_2025_20k_unorm/album_RNO4_train.hdf5'
test_album_path = '/data/i3store/users/ssued/albums/RNO_album_11_03_2025_20k_unorm/album_RNO4_test.hdf5'

# Load Datasets

# Might want to add these to skip compounding statistics:
# For training: [1.8211910724639893,1229.3675537109375,15.867149353027344,1235.3389892578125,-853.52490234375,446.32781982421875]
# For testing:  [-4.812802314758301,1236.3431396484375,14.210803985595703,1237.897216796875,-850.2981567382812,443.872314453125]

# album = AlbumDataset(album_path, transform=None, target_transform=None)
train_album = AlbumDataset(train_album_path, target_transform=TARGET_TRANSFORM, normalize_labels=True)
test_album = AlbumDataset(test_album_path, target_transform=TARGET_TRANSFORM, normalize_labels=True)

print(f'Train album size: {train_album.num_images} | Test album size: {test_album.num_images}')

# Load DataLoaders
train_data_loader = DataLoader(dataset = train_album,
                               batch_size = BATCH_SIZE,
                               shuffle = True,
                               num_workers = 16,
                               pin_memory=True)
test_data_loader = DataLoader(dataset = test_album,
                              batch_size = BATCH_SIZE,
                              shuffle = False,
                              num_workers = 4,
                              pin_memory=True)

print(f'Number of train batches: {len(train_data_loader)} | Number of test batches: {len(test_data_loader)}')

# Initialize model
model = models.RNO_four_late_non_linear_merge(input_shape=1,
                          hidden_units=HIDDEN_UNITS,
                          output_shape=3,
                          num_epochs=NUM_EPOCHS,
                          batch_size=BATCH_SIZE,
                          num_train_batches=len(train_data_loader),
                          leak_factor=LEAK_FACTOR,
                          dropout_rate=DROPOUT_RATE,
                          temporal_res=TEMPORAL_RES
                          )

# Setup optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr = LEARNING_RATE)
optimizer_name = optimizer.__class__.__name__
# Setup loss function
#loss_fn = torch.nn.HuberLoss(delta=50)
loss_fn = torch.nn.MSELoss()
loss_fn_name = loss_fn.__class__.__name__

experiment_name = (f'exp_{model.__class__.__name__}' +
                  f'_bn{BATCH_SIZE}' +
                  f'_tr{len(train_data_loader)}' +
                  f'_te{len(test_data_loader)}' +
                  f'_lfn-{loss_fn_name}' +
                  f'_opt-{optimizer_name}' +
                  f'_hiddenu-{HIDDEN_UNITS}' +
                  f'_lr-{LEARNING_RATE}' +
                  f'_lFactor-{LEAK_FACTOR}' +
                #   f'_batchnorm-{NORMALIZE_INPUTS}' +
                  f'_cartesian_transform-{CARTESIAN}' +
                  f'_temporalRes-{TEMPORAL_RES}' +
                ''
               )

# Create experiments directory if it doesn't exist
os.makedirs('experiments', exist_ok=True)
experiment_path = os.path.join('experiments', experiment_name)

# Warn user if experiment already exists and wait for confirmation
if os.path.exists(experiment_path):
    print('WARNING: Experiment with this name already exists. Run data will be saved to this existing directory. ' \
    'To continue create a file titled "y" in the experiment directory',flush=True)
    start = time.time()
    timeout = 60  # seconds
    while True:
        if os.path.exists(os.path.join(experiment_path, 'y')):
            print('"y" file spotted, continuing...',flush=True)
            os.remove(os.path.join(experiment_path, 'y'))
            break
        if time.time() - start >= timeout:
            print('Timeout reached (60s), exiting program...')
            sys.exit(1)
        time.sleep(1)
else:
    os.makedirs(experiment_path, exist_ok=True)


# Setup logging
logger = logging.getLogger('experiment_log') # Setup logging
logging.basicConfig(filename=f'{experiment_path}/experiment.log',
                    filemode='w',
                    level=logging.INFO,
                    format='[%(levelname)s: %(asctime)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger.info(f"Starting experiment: {experiment_name}")
logger.info(f"Device: {device}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Model: {type(model).__name__}")

logger.info(f"Optimizer: {optimizer_name}")
logger.info(f"Loss function: {loss_fn_name}")


train_test(model = model, 
           train_dataloader = train_data_loader, 
           test_dataloader = test_data_loader, 
           optimizer = optimizer,
           scheduler = None,
           loss_fn = loss_fn,
           device = device,
           experiment_name = experiment_name,
           epochs = NUM_EPOCHS,
           checkpoint_freq = CHECKPOINT_FREQ,
           checkpoint_path = checkpoint_path,
           loss_file = 'losses.txt',
           logger = logger,
           wandb_id=WANDB_ID)
