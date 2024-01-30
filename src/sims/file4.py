import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyactup
import sys
sys.path.insert(1,'/Users/mo6xj/Projects/ibl-can/src/func_pac')
import ibl_can_func_pac as ibl

'''PRELIMINARIES'''
data_in_dir = '/Users/mo6xj/Projects/ibl-can/data-in/'
data_out_dir = '/Users/mo6xj/Projects/ibl-can/data-out/'
graph_out_dir = '/Users/mo6xj/Projects/ibl-can/graph-out/'
#THE ORIGINAL DATA FROM ANES 2012 10 Items
#df = pd.read_csv(f'{data_in_dir}Obama.csv')


#USABLE FOR PUBLICATION
#IBL
noise_list = [.01,.1,.20,.30,.40,.5,.6,.7,.8,.9,1,1.5,2,4,6,8,10]
catch_endstates = []
for i in noise_list:
    #INITS
    ns_fn = i
    sim_name = f'ibl_noise_{ns_fn}'
    #BEGIN ANALYSIS
    sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    catch_endstates.append(sim_endstates)

#WRITE FOR COMPARE TO ISING SAMPLER SIMs   
ibl_endstates = catch_endstates

#CAN - ISING SAMPLER SIMS
#NOTE CAN SIMS RUN ALREADY IN R
noise_list = [20,10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01]
catch_endstates = []
for i in noise_list:
    #INITS
    ns_fn = i
    sim_name = f'can_beta_{ns_fn}'
    #BEGIN ANALYSIS
    sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    catch_endstates.append(sim_endstates)
    
can_endstates = catch_endstates

'''NOW WHAT DO WE DO WITH THESE'''

ibl_noise_list = [.01,.1,.20,.30,.40,.5,.6,.7,.8,.9,1,1.5,2,4,6,8,10]
can_noise_list = [20,10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01]
for i in range(0,17):
    print('NEW LOOP')
    print('NEW LOOP')
    print('NEW LOOP')
    print('NOISE INDEX: ', i)
    print('IBL, noise: ',ibl_noise_list[i])
    print(ibl_endstates[i])
    print('CAN, noise: ',can_noise_list[i])
    print(can_endstates[i]) 

#EOF