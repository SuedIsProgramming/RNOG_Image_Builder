import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all functions from utils_dir (handled by __init__.py)
from utils_dir import *

def obtain_experiment_path_by_idx(experiments_dir: str, idx: int):
    experiments = [f for f in os.listdir(experiments_dir) if not f.startswith('.')]
    print(f'Experiment at id {idx}: {experiments[idx]}')
    return os.path.join(experiments_dir,experiments[idx])
    
def load_model_from_checkpoint(input_shape: int, hidden_units: int, output_shape: int, checkpoint = None):
    
    if checkpoint:
        checkpoint_state_dict = torch.load(checkpoint)['model_state_dict']
        
        model = VertexFinder1_0_0(input_shape=input_shape,
                                  hidden_units=hidden_units, 
                                  output_shape=output_shape)

        model.load_state_dict(torch.load(checkpoint_state_dict))
    else:
        model = VertexFinder1_0_0(input_shape=input_shape,
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