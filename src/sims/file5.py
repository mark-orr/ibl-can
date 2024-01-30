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
ns = 0.5 #noise
cue_list = ['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd']

for i in cue_list:
    #INITS
    cue_name = i
    num_retrieve = 100
    sim_name = f'ibl_cue_{cue_name}'
    #BEGIN PYACTUP SIM
    m = pyactup.Memory(noise=ns)
    ibl.load_m(m,df)
    m.advance()
    sim = ibl.sim_w_cue(m,num_retrieve,i)#the numpy sim output array
    #WRIT SIM TO DATA
    df_sim = pd.DataFrame(sim,columns=df.columns)
    df_sim.to_csv(f'{data_out_dir}{sim_name}.csv',index=None)

'''SIM FOR CAN DONE IN R IN FILE1.R'''

'''ANALYSIS'''
#WASS ANALYSIS 
#IBL
#GRAB BASE DIST
sim_name = 'ibl_noise_0.5' 
base_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
base_dist_endstates = ibl.dist_of_endstates(base_sim_endstates)

#COMPS
cue_list = ['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd']
catch_wass = np.array([])
for i in cue_list:
    #INITS
    cue_name = i
    sim_name = f'ibl_cue_{cue_name}'
    #GRAB SIM ENDSTATES FOR CUE
    test_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    test_dist_endstates = ibl.dist_of_endstates(test_sim_endstates)
    #COMPUTE WASS AND RECORD
    catch_wass = np.append(catch_wass,ibl.wass_norm(base_dist_endstates,test_dist_endstates))

ibl_catch_cue_wass = catch_wass

#CAN
#GRAB BASE DIST
sim_name = 'can_beta_2' 
base_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
base_dist_endstates = ibl.dist_of_endstates(base_sim_endstates)

#COMPS
cue_list = ['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd']
catch_wass = np.array([])
for i in cue_list:
    #INITS
    cue_name = i
    sim_name = f'can_cue_{cue_name}'
    #GRAB SIM ENDSTATES FOR CUE
    test_sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    test_dist_endstates = ibl.dist_of_endstates(test_sim_endstates)
    #COMPUTE WASS AND RECORD
    catch_wass = np.append(catch_wass,ibl.wass_norm(base_dist_endstates,test_dist_endstates))

can_catch_cue_wass = catch_wass

#COMPARISION PLOT
plt.style.use('fivethirtyeight')
x = np.arange(0,10)
fig, axes = plt.subplots(1,1,figsize=(5,5))

axes.plot(x,can_catch_cue_wass,color='blue',marker='o',dashes=[0,2,2,2],label='CAN',linewidth=1)
axes.plot(x,ibl_catch_cue_wass,color='black',marker='+',dashes=[0,0,2,2],label='IBL',linewidth=1)
axes.legend()

axes.set_xlabel('Retrieval Cue', labelpad=2,size=10)
axes.set_ylabel('Wass. Distance', labelpad=1,size=10)
axes.set_ylim([0,1])
axes.tick_params(axis='x', labelsize=7)
axes.tick_params(axis='y', labelsize=7)
axes.set_xticks(x)
axes.set_xticklabels(labels=['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd'])
plt.savefig(f'{graph_out_dir}Cued_WASS_CAN-IBL.png',dpi=300,bbox_inches='tight')
#plt.show()


#PLOTS OF GIBBS FOR EACH CUE
#IBL
fig, axes = plt.subplots(2,5,figsize=(5,5),sharex=True,sharey=True)

cue_list = ['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd']

counter = 0
for x in range(0,2):
    for y in range(0,5):
        cue_name = cue_list[counter]
        sim_name = f'ibl_cue_{cue_name}'
        sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
        dist_endstates = ibl.dist_of_endstates(sim_endstates)
        axes[x,y].plot(ibl.dist_of_endstates(sim_endstates))
        axes[x,y].set_ylim(0,100)
        
        axes[x,y].text(500,90,cue_list[counter],fontsize=7)
        counter = counter+1
        #print(x,' ',y,' ',counter)

axes[0,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,1].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,2].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,3].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,4].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)

axes[1,0].tick_params(axis='x', labelsize=5)
axes[1,1].tick_params(axis='x', labelsize=5)
axes[1,2].tick_params(axis='x', labelsize=5)
axes[1,3].tick_params(axis='x', labelsize=5)
axes[1,4].tick_params(axis='x', labelsize=5)
axes[0,0].tick_params(axis='y', labelsize=7)
axes[1,0].tick_params(axis='y', labelsize=7)
axes[1,0].set_xticks([10,1013])
axes[1,0].set_xticklabels(labels=['10','1013'])

plt.savefig(f'{graph_out_dir}IBL_Gibbs_Cued_2X5.png',dpi=300,bbox_inches='tight')
#plt.show()


#CAN
fig, axes = plt.subplots(2,5,figsize=(5,5),sharex=True,sharey=True)

cue_list = ['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd']

counter = 0
for x in range(0,2):
    for y in range(0,5):
        cue_name = cue_list[counter]
        sim_name = f'can_cue_{cue_name}'
        sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
        dist_endstates = ibl.dist_of_endstates(sim_endstates)
        axes[x,y].plot(ibl.dist_of_endstates(sim_endstates))
        axes[x,y].set_ylim(0,100)
        
        axes[x,y].text(500,90,cue_list[counter],fontsize=7)
        counter = counter+1
        #print(x,' ',y,' ',counter)

axes[0,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,1].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,2].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,3].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,4].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)

axes[1,0].tick_params(axis='x', labelsize=5)
axes[1,1].tick_params(axis='x', labelsize=5)
axes[1,2].tick_params(axis='x', labelsize=5)
axes[1,3].tick_params(axis='x', labelsize=5)
axes[1,4].tick_params(axis='x', labelsize=5)
axes[0,0].tick_params(axis='y', labelsize=7)
axes[1,0].tick_params(axis='y', labelsize=7)
axes[1,0].set_xticks([10,1013])
axes[1,0].set_xticklabels(labels=['10','1013'])

plt.savefig(f'{graph_out_dir}CAN_Gibbs_Cued_2X5.png',dpi=300,bbox_inches='tight')
#plt.show()


#EOF