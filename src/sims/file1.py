import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/Users/mo6xj/Projects/ibl-can/src/func_pac')
import ibl_can_func_pac as ibl

'''PRELIMINARIES'''
data_in_dir = '../../data-in'
df = pd.read_csv('Obama.csv')


'''SIM RUNS'''
catch_num_attractors = np.array([])
for i in noise_list:
    ns = i
    ns_fn = i
    num_retrieve = 100
    sim_name = f'ibl_noise_{ns_fn}'
    m = pyactup.Memory(noise=ns)
    load_m(m,df)
    m.advance()
    sim = sim_no_cue(m,num_retrieve)#the numpy sim output array
    df_sim = pd.DataFrame(sim,columns=df.columns)
    df_sim.to_csv(f'{sim_name}.csv',index=None)

    sim_endstates = extract_endstates(f'{sim_name}.csv')

    catch_num_attractors = np.append(catch_num_attractors,len(sim_endstates[0]))

    plt.style.use('fivethirtyeight')
    imgplot = plt.imshow(extract_endstates_for_plot(sim_endstates)[1:],cmap="binary",aspect='auto',interpolation='none') 
    plt.xlabel('State Vector Index')
    plt.ylabel('Point-Attractor Index')
    plt.colorbar()
    plt.savefig(f'Fig_Endstates_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()

    plt.plot(dist_of_endstates(sim_endstates))
    plt.savefig(f'Fig_Gibbs_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()
    
ibl_num_att_by_noise = catch_num_attractors

#CAN
noise_list = [20,10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01]
catch_num_attractors = np.array([])
for i in noise_list:
    ns = i
    ns_fn = i
    num_retrieve = 100
    sim_name = f'can_beta_{ns_fn}'
    
    sim_endstates = extract_endstates(f'{sim_name}.csv')

    catch_num_attractors = np.append(catch_num_attractors,len(sim_endstates[0]))
    plt.style.use('fivethirtyeight')
    imgplot = plt.imshow(extract_endstates_for_plot(sim_endstates)[1:],cmap="binary",aspect='auto',interpolation='none') 
    plt.xlabel('State Vector Index')
    plt.ylabel('Point-Attractor Index')
    plt.colorbar()
    plt.savefig(f'Fig_Endstates_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()

    plt.plot(dist_of_endstates(sim_endstates))
    plt.savefig(f'Fig_Gibbs_{sim_name}.png',dpi=400,bbox_inches='tight')
    #plt.show()
    plt.clf()
    

can_num_att_by_noise = catch_num_attractors

#COMPARISION PLOT
plt.plot(can_num_att_by_noise)
plt.plot(ibl_num_att_by_noise)
plt.show()
