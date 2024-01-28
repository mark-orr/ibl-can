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

#IBL
#GRAB BASE DIST AS LOWEST NOISE
ns = .01
ns_fn = ns
num_retrieve = 100
sim_name = f'ibl_noise_{ns_fn}'
#read in lowest noise file
base_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
base_dist_endstates = ibl.dist_of_endstates(base_sim_endstates)

noise_list = [.1,.20,.30,.40,.5,.6,.7,.8,.9,1,1.5,2,4,6,8,10]
catch_wass = np.array([])
for i in noise_list:
    ns_fn = i
    sim_name = f'ibl_noise_{ns_fn}'
    test_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    test_dist_endstates = ibl.dist_of_endstates(test_sim_endstates)

    catch_wass = np.append(catch_wass,ibl.wass_norm(base_dist_endstates,test_dist_endstates))

ibl_catch_wass = catch_wass


#CAN
#GRAB BASE DIST AS LOWEST NOISE
ns = 20
ns_fn = ns
num_retrieve = 100
sim_name = f'can_beta_{ns_fn}'
#read in lowest noise file
base_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
base_dist_endstates = ibl.dist_of_endstates(base_sim_endstates)

noise_list = [10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01]
catch_wass = np.array([])
for i in noise_list:
    ns_fn = i
    sim_name = f'can_beta_{ns_fn}'
    test_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    test_dist_endstates = ibl.dist_of_endstates(test_sim_endstates)

    catch_wass = np.append(catch_wass,ibl.wass_norm(base_dist_endstates,test_dist_endstates))

can_catch_wass = catch_wass


#COMPARE BOTH
plt.style.use('fivethirtyeight')
plt.plot(ibl_catch_wass)
plt.plot(can_catch_wass)
plt.savefig(f'{graph_out_dir}Wass_CAN-IBL.png',dpi=300,bbox_inches='tight')
plt.show()



#EOF