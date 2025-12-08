from h5py import File
import numpy as np
import os
import shutil

def copy_event(album_source, album_dest_path, event_idx):
    """
    Copies a specific event group from one HDF5 file to another.
    Creates the destination file if it doesn't exist.
    """
    
    # Ensure the destination directory exists
    dest_dir = os.path.dirname(album_dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Use 'a' (append) for dest. 'w' would overwrite/delete the existing file!
    with File(album_source, 'r') as source_album, \
         File(album_dest_path, 'a') as dest_album:

        event_key = f'event{event_idx}'

        # 1. Check if the event actually exists in source
        if event_key not in source_album.keys():
            print(f"Error: {event_key} not found in {album_source}")
            return

        # 2. Check if event exists in destination to prevent collision errors
        if event_key in dest_album:
            print(f"Warning: {event_key} already exists in destination. Overwriting.")
            del dest_album[event_key]

        # 3. Perform the copy
        # h5py's .copy() method handles recursive copying of groups and datasets
        source_album.copy(event_key, dest_album)
        
        print(f"Successfully copied {event_key} to {album_dest_path}")

def trainTest_split(album_dir,album_name,train_ratio = 0.8, seed = 42, backup= True):

    base_name = os.path.splitext(album_name)[0]
    with File(os.path.join(album_dir,album_name), 'r') as album, \
        File(os.path.join(album_dir,f'{base_name}_train.hdf5'), 'w') as train_album, \
        File(os.path.join(album_dir,f'{base_name}_test.hdf5'), 'w') as test_album:

        if backup:
            backup_path = os.path.join(album_dir, f"backup_{album_name}")
            print(f'Backing up album to {backup_path}')
            if not os.path.exists(backup_path):
                shutil.copy2(os.path.join(album_dir, album_name), backup_path)

        # Get all event keys and create random split
        all_keys = list(album.keys())
        total_size = len(all_keys)
        split_index = int(total_size * train_ratio)

        np.random.seed(seed)
        shuffled_indices = np.random.permutation(total_size)

        train_indices = shuffled_indices[:split_index]
        test_indices = shuffled_indices[split_index:]

        # Copy training data
        print('Copying training data...')
        for i, orig_idx in enumerate(train_indices):
            print(f'Copying training sample {i+1}/{len(train_indices)} (original index {orig_idx})')
            orig_key = all_keys[orig_idx]
            new_key = f'event{i+1}'  # Reindex starting from 1
            
            train_album.create_group(new_key)
            train_album[new_key]['image'] = album[orig_key]['image'][:]
            train_album[new_key]['label'] = album[orig_key]['label'][:]
        
        # Copy test data  
        for i, orig_idx in enumerate(test_indices):
            print(f'Copying testing sample {i+1}/{len(test_indices)} (original index {orig_idx})')
            orig_key = all_keys[orig_idx]
            new_key = f'event{i+1}'  # Reindex starting from 1
            
            test_album.create_group(new_key)
            test_album[new_key]['image'] = album[orig_key]['image'][:]
            test_album[new_key]['label'] = album[orig_key]['label'][:]
        
        print(f"Split complete: {len(train_indices)} training, {len(test_indices)} test samples")

def swap_phi_theta(album_path):
    with File(album_path, 'r+') as album:
        num_events = len(album.keys())
        for idx in range(num_events):
            print(f'\rSwapping phis and thetas... ({idx+1}/{num_events})', end='',flush=True)
            event_key = f'event{idx+1}'
            
            # Read the data
            label = album[event_key]['label'][:]
            r, theta, phi = label
            
            # Modify in place
            album[event_key]['label'][:] = [r, phi, theta]
        print('\nDone!')