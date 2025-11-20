#!/data/condor_shared/users/ssued/RNOG_Image_Builder/venv/bin/python3

"""
HDF5 Album Consolidation Script

This script consolidates multiple individual HDF5 album files into a single master album file.
It processes all HDF5 files in the 'albums' directory (except the master 'album.hdf5' file)
and copies their contents into the master file with sequential event numbering.

The script maintains proper event indexing by finding the highest existing event number
in the master file and continuing the sequence from there.

Dependencies:
   - h5py: For HDF5 file operations
   - utils.my_utils: Custom utility functions for HDF5 index management

File Structure:
   albums/
   ├── album.hdf5          # Master consolidated album file (destination)
   ├── mini_album_1.hdf5   # Individual album files (sources)
   ├── mini_album_2.hdf5
   └── ...

Usage:
   Run this script from the main directory.
   The script will automatically process all HDF5 files and consolidate them.
"""
import sys # Cannot escape this import :C

sys.path.append('/data/condor_shared/users/ssued/RNOG_Image_Builder')

from utils.my_utils import find_max_hdf5_index, get_album_name
from pathlib import Path
import time
import h5py
import os

root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder
start_time = time.time()

# Configuration
album_name = get_album_name()
albums_path = f'/data/i3store/users/ssued/albums/{album_name}/n3_album'
master_album_filename = 'n3_album.hdf5'

# Process each individual album file in the albums directory
for albums_filename in os.listdir(albums_path):
    # Skip the master album file to avoid processing it into itself
    if albums_filename != master_album_filename:
        # Construct file paths
        source_album_path = f'{albums_path}/{albums_filename}'
        master_album_path = f'{albums_path}/{master_album_filename}'
        
        # Open both source and destination HDF5 files
        # 'r' mode: read-only access to source file
        # 'a' mode: append/read-write access to master file (creates if doesn't exist)
        with h5py.File(source_album_path, 'r') as h5_source, \
            h5py.File(master_album_path, 'a') as h5_destination:
            
            print(f'Copying events from {h5_source} to {h5_destination}')

            # Find the current maximum event index in the master file
            # This ensures we continue numbering from where we left off
            max_index = find_max_hdf5_index(h5_destination)
            
            # Copy each dataset/group from the source file to the master file
            for dataset_key in h5_source.keys():
                # Copy the dataset with a new sequential event name
                # Event numbering starts from max_index + 1 and increments
                new_event_name = f'event{max_index + 1}'
                h5_source.copy(h5_source[dataset_key], h5_destination, new_event_name)
                max_index += 1
    
        try:
            pass
            os.remove(source_album_path)
            print(f"Deleted: {source_album_path}")
        except OSError as e:
            print(f"Error deleting {source_album_path}: {e}")

print("Album consolidation completed successfully!")
end_time = time.time()
print(f'Runtime : {end_time - start_time} s')
# Remove left over .hdf5 .nur files from jobs_simulation_data

# for file in os.listdir(f'{root_dir}/jobs/jobs_simulation_data'):
#     if file.endswith('.hdf5') or file.endswith('.nur'):
#         file_path = os.path.join(f'{root_dir}/jobs/jobs_simulation_data', file)
#         os.remove(file_path)
        #print(f"Deleted: {file}")
