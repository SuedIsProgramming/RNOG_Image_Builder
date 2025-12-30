import sys
import os

# Configuration
sim_num = int(50_000) 
root_dir = os.getcwd() 
jobs_path = f'{root_dir}/jobs'
dag_filename = "dagman.dag"
dag_path = f"{root_dir}/jobs/{dag_filename}"

# Paths to submit files
step1_path = f'{jobs_path}/step1.sub'
step2_path = f'{jobs_path}/step2.sub'
step3_path = f'{jobs_path}/step3.sub'
step3_multip_path = f'{jobs_path}/step3_multip.sub'
step4_path = f'{jobs_path}/step4.sub'
step4_multip1_path = f'{jobs_path}/step4_n1.sub'
step4_multip2_path = f'{jobs_path}/step4_n2.sub'
step4_multip3_path = f'{jobs_path}/step4_n3.sub'
step4_multip4_path = f'{jobs_path}/step4_n4.sub'

print(f'Generating DAG for {sim_num} simulations...')
print(f'Will save to {dag_path}')

with open(dag_path, 'w') as f:
    
    # 1. Config Header
    f.write('CONFIG jobs/dagman.config \n\n')

    # Handle the trivial single-job case optimization
    if sim_num == 1:
        # Step 0
        i = 0
        f.write(f'JOB stepOne_{i} {step1_path} \n')
        f.write(f'VARS stepOne_{i} sim_num="{i}" \n')
        f.write(f'JOB stepTwo_{i} {step2_path} \n')
        f.write(f'VARS stepTwo_{i} sim_num="{i}" \n')
        f.write(f'JOB stepThree_multip_{i} {step3_multip_path} \n')
        f.write(f'VARS stepThree_multip_{i} sim_num="{i}" \n\n')
        f.write(f'PARENT stepOne_{i} CHILD stepTwo_{i}\n')
        f.write(f'PARENT stepTwo_{i} CHILD stepThree_multip_{i}\n\n')
        
        f.write(f'JOB stepFour {step4_path} \n')
        f.write('PARENT stepThree_multip_0 CHILD stepFour\n')
        print("Done (Single Job).")
        sys.exit()

    # 2. Main Simulation Loop (Generates ALL 0 to N-1 jobs)
    for i in range(sim_num):
        f.write(f'JOB stepOne_{i} {step1_path} \n')
        f.write(f'VARS stepOne_{i} sim_num="{i}" \n')
        f.write(f'JOB stepTwo_{i} {step2_path} \n')
        f.write(f'VARS stepTwo_{i} sim_num="{i}" \n')
        f.write(f'JOB stepThree_multip_{i} {step3_multip_path} \n')
        f.write(f'VARS stepThree_multip_{i} sim_num="{i}" \n\n')
        
        f.write(f'PARENT stepOne_{i} CHILD stepTwo_{i} \n')
        f.write(f'PARENT stepTwo_{i} CHILD stepThree_multip_{i} \n\n')

    # 3. Step 4 (Aggregation)
    # Define the aggregation jobs
    f.write(f'JOB stepFour_multip_1 {step4_multip1_path}\n')
    f.write(f'JOB stepFour_multip_2 {step4_multip2_path}\n')
    f.write(f'JOB stepFour_multip_3 {step4_multip3_path}\n')
    f.write(f'JOB stepFour_multip_4 {step4_multip4_path}\n\n')

    for j in range(sim_num):
        f.write(f'PARENT stepThree_multip_{j} CHILD stepFour_multip_1 \n')

    # Link the aggregation steps sequentially
    f.write('PARENT stepFour_multip_1 CHILD stepFour_multip_2 \n')
    f.write('PARENT stepFour_multip_2 CHILD stepFour_multip_3 \n')
    f.write('PARENT stepFour_multip_3 CHILD stepFour_multip_4 \n\n')

    # Final cleanup step
    f.write(f'FINAL stepFour {step4_path}\n')

print("DAG file generation complete.")