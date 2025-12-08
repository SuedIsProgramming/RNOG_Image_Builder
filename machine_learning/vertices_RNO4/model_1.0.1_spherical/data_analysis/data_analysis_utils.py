import numpy as np
import math
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))

# Import all functions from utils_dir (handled by __init__.py)
from utils_dir import *

def polar2cart(coord):
    r, phi, theta = coord
    
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

def normalized_polar2cart(coord,dataset):
    r, phi, theta = coord
    r = r * dataset.get_rnorm()
    phi = phi*2*np.pi
    theta = theta*2*np.pi
    
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
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

            x_targ,y_targ,z_targ = polar2cart(target)
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