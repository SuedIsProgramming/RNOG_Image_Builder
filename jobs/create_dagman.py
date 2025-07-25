import sys
import os


#  Taken from https://github.com/umd-pa/tutorials-icecube/blob/main/convert_i3_to_hdf5/step2/step2a_make_dag.py

# This script creates a DAG file for the condor job scheduler. The DAG file is a text file that specifies the order in which jobs should be run.

sim_num = 2**4 # Number of simulations to run 

root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder

jobs_path = f'{root_dir}/jobs'

step1_path = f'{jobs_path}/step1.sub'
step2_path = f'{jobs_path}/step2.sub'
step3_path = f'{jobs_path}/step3.sub'
step4_path = f'{jobs_path}/step4.sub'

# now we have to write the dag file itself
dag_filename = f"dagman.dag"
instructions = ""
root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder
dag_path = f"{root_dir}/jobs/{dag_filename}"

#instructions += 'CONFIG config.dagman \n' Im not sure this is necessary

print(f'Will save to {dag_path}')

instructions += f'CONFIG jobs/dagman.config \n\n'
instructions += f'JOB stepOne_0 {step1_path} \n'
instructions += f'VARS stepOne_0 sim_num="0" \n'
instructions += f'JOB stepTwo_0 {step2_path} \n'
instructions += f'VARS stepTwo_0 sim_num="0" \n'
instructions += f'JOB stepThree_0 {step3_path} \n'
instructions += f'VARS stepThree_0 sim_num="0" \n\n'
instructions += f'PARENT stepOne_0 CHILD stepTwo_0\n'
instructions += f'PARENT stepTwo_0 CHILD stepThree_0\n\n'

if sim_num == 1:
    instructions += f'JOB stepFour {step4_path} \n'
    instructions += f'PARENT stepThree_0 CHILD stepFour'
    with open(dag_path, 'w') as fwrite:
        fwrite.write(instructions)
    sys.exit()

sim_num = sim_num - 1 # shh, dont tell anyone

for i in range(1, sim_num):
    instructions += f'JOB stepOne_{i} {step1_path} \n'
    instructions += f'VARS stepOne_{i} sim_num="{i}" \n'
    instructions += f'JOB stepTwo_{i} {step2_path} \n'
    instructions += f'VARS stepTwo_{i} sim_num="{i}" \n'
    instructions += f'JOB stepThree_{i} {step3_path} \n'
    instructions += f'VARS stepThree_{i} sim_num="{i}" \n\n'
    instructions += f'PARENT stepOne_{i} CHILD stepTwo_{i} \n'
    instructions += f'PARENT stepTwo_{i} CHILD stepThree_{i} \n\n'

instructions += f'JOB stepOne_{sim_num} {step1_path} \n'
instructions += f'VARS stepOne_{sim_num} sim_num="{sim_num}" \n'
instructions += f'JOB stepTwo_{sim_num} {step2_path} \n'
instructions += f'VARS stepTwo_{sim_num} sim_num="{sim_num}" \n'
instructions += f'JOB stepThree_{sim_num} {step3_path} \n'
instructions += f'VARS stepThree_{sim_num} sim_num="{sim_num}" \n\n'
instructions += f'PARENT stepOne_{sim_num} CHILD stepTwo_{sim_num}\n'
instructions += f'PARENT stepTwo_{sim_num} CHILD stepThree_{sim_num}\n\n'

instructions += f'FINAL stepFour {step4_path}\n'

with open(dag_path, 'w') as fwrite:
    fwrite.write(instructions)
