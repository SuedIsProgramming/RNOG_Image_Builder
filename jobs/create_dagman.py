import sys
import os


#  Taken from https://github.com/umd-pa/tutorials-icecube/blob/main/convert_i3_to_hdf5/step2/step2a_make_dag.py

# This script creates a DAG file for the condor job scheduler. The DAG file is a text file that specifies the order in which jobs should be run.

sim_num = int() # Number of simulations to run 

# 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64, 2^7=128, 2^8=256, 2^9=512, 2^10=1024, 2^11=2048, 2^12=4096, 2^13=8192, 2^14=16384, 2^15=24576, 2^16=32768


root_dir = os.getcwd() # Should be main directory: RNOG_Image_Builder

jobs_path = f'{root_dir}/jobs'

step1_path = f'{jobs_path}/step1.sub'
step2_path = f'{jobs_path}/step2.sub'
step3_path = f'{jobs_path}/step3.sub'
step3_multip_path = f'{jobs_path}/step3_multip.sub'
step4_path = f'{jobs_path}/step4.sub'
step4_multip1_path = f'{jobs_path}/step4_n1.sub'
step4_multip2_path = f'{jobs_path}/step4_n2.sub'
step4_multip3_path = f'{jobs_path}/step4_n3.sub'
step4_multip4_path = f'{jobs_path}/step4_n4.sub'

# now we have to write the dag file itself
dag_filename = f"dagman_debug.dag"
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
instructions += f'JOB stepThree_multip_0 {step3_multip_path} \n'
instructions += f'VARS stepThree_multip_0 sim_num="0" \n\n'
instructions += f'PARENT stepOne_0 CHILD stepTwo_0\n'
instructions += f'PARENT stepTwo_0 CHILD stepThree_multip_0\n\n'

if sim_num == 1:
    instructions += f'JOB stepFour {step4_path} \n'
    instructions += f'PARENT stepThree_multip_0 CHILD stepFour'
    with open(dag_path, 'w') as fwrite:
        fwrite.write(instructions)
    sys.exit()

sim_num = sim_num - 1 # shh, dont tell anyone

for i in range(1, sim_num):
    instructions += f'JOB stepOne_{i} {step1_path} \n'
    instructions += f'VARS stepOne_{i} sim_num="{i}" \n'
    instructions += f'JOB stepTwo_{i} {step2_path} \n'
    instructions += f'VARS stepTwo_{i} sim_num="{i}" \n'
    instructions += f'JOB stepThree_multip_{i} {step3_multip_path} \n'
    instructions += f'VARS stepThree_multip_{i} sim_num="{i}" \n\n'
    instructions += f'PARENT stepOne_{i} CHILD stepTwo_{i} \n'
    instructions += f'PARENT stepTwo_{i} CHILD stepThree_multip_{i} \n\n'

instructions += f'JOB stepOne_{sim_num} {step1_path} \n'
instructions += f'VARS stepOne_{sim_num} sim_num="{sim_num}" \n'
instructions += f'JOB stepTwo_{sim_num} {step2_path} \n'
instructions += f'VARS stepTwo_{sim_num} sim_num="{sim_num}" \n'
instructions += f'JOB stepThree_multip_{sim_num} {step3_multip_path} \n'
instructions += f'VARS stepThree_multip_{sim_num} sim_num="{sim_num}" \n\n'
instructions += f'PARENT stepOne_{sim_num} CHILD stepTwo_{sim_num}\n'
instructions += f'PARENT stepTwo_{sim_num} CHILD stepThree_multip_{sim_num}\n\n'

instructions += f'JOB stepFour_multip_1 {step4_multip1_path}\n'
instructions += f'JOB stepFour_multip_2 {step4_multip2_path}\n'
instructions += f'JOB stepFour_multip_3 {step4_multip3_path}\n'
instructions += f'JOB stepFour_multip_4 {step4_multip4_path}\n\n'

instructions += f'PARENT stepThree_multip_{sim_num} CHILD stepFour_multip_1 \n'
instructions += f'PARENT stepFour_multip_1 CHILD stepFour_multip_2 \n'
instructions += f'PARENT stepFour_multip_2 CHILD stepFour_multip_3 \n'
instructions += f'PARENT stepFour_multip_3 CHILD stepFour_multip_4 \n\n'

instructions += f'FINAL stepFour {step4_path}\n'

with open(dag_path, 'w') as fwrite:
    fwrite.write(instructions)
