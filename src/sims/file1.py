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
df = pd.read_csv(f'{data_in_dir}Obama.csv')

'''SIM RUNS'''
#IBL PYACTUP
noise_list = [.01,.1,.20,.30,.40,.5,.6,.7,.8,.9,1,1.5,2,4,6,8,10]
catch_num_attractors = np.array([])
for i in noise_list:
    #INITS
    ns = i
    ns_fn = i
    num_retrieve = 100
    sim_name = f'ibl_noise_{ns_fn}'
    #BEGIN PYACTUP SIM
    m = pyactup.Memory(noise=ns)
    ibl.load_m(m,df)
    m.advance()
    sim = ibl.sim_no_cue(m,num_retrieve)#the numpy sim output array
    #WRIT SIM TO DATA
    df_sim = pd.DataFrame(sim,columns=df.columns)
    df_sim.to_csv(f'{data_out_dir}{sim_name}.csv',index=None)
    #BEGIN ANALYSIS
    sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    catch_num_attractors = np.append(catch_num_attractors,len(sim_endstates[0]))
    plt.style.use('fivethirtyeight')
    imgplot = plt.imshow(ibl.extract_endstates_for_plot(sim_endstates)[1:],cmap="binary",aspect='auto',interpolation='none') 
    plt.xlabel('State Vector Index')
    plt.ylabel('Point-Attractor Index')
    plt.colorbar()
    plt.savefig(f'{graph_out_dir}Fig_Endstates_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.plot(ibl.dist_of_endstates(sim_endstates))
    plt.savefig(f'{graph_out_dir}Fig_Gibbs_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()
#WRITE FOR COMPARE TO ISING SAMPLER SIMs   
ibl_num_att_by_noise = catch_num_attractors

#CAN - ISING SAMPLER SIMS
#NOTE CAN SIMS RUN ALREADY IN R
noise_list = [20,10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01]
catch_num_attractors = np.array([])
for i in noise_list:
    #INITS
    ns = i
    ns_fn = i
    num_retrieve = 100
    sim_name = f'can_beta_{ns_fn}'
    #BEGIN ANALYSIS
    sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    catch_num_attractors = np.append(catch_num_attractors,len(sim_endstates[0]))
    plt.style.use('fivethirtyeight')
    imgplot = plt.imshow(ibl.extract_endstates_for_plot(sim_endstates)[1:],cmap="binary",aspect='auto',interpolation='none') 
    plt.xlabel('State Vector Index')
    plt.ylabel('Point-Attractor Index')
    plt.colorbar()
    plt.savefig(f'{graph_out_dir}Fig_Endstates_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.plot(ibl.dist_of_endstates(sim_endstates))
    plt.savefig(f'{graph_out_dir}Fig_Gibbs_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()
    

can_num_att_by_noise = catch_num_attractors

#COMPARISION PLOT
plt.plot(can_num_att_by_noise)
plt.plot(ibl_num_att_by_noise)
plt.savefig(f'{graph_out_dir}Num_Endstates_CAN-IBL.png',dpi=300,bbox_inches='tight')
plt.show()
