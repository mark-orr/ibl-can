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
noise_list = [.01,.2,.4,.8,1,1.5,4,10]
fig, axes = plt.subplots(2,4,figsize=(3,5),sharex=True,sharey=True)

counter = 0

for x in range(0,2):
    for y in range(0,4):
        ns_fn = noise_list[counter]
        sim_name = f'ibl_noise_{ns_fn}'
        sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
        dist_endstates = ibl.dist_of_endstates(sim_endstates)
        axes[x,y].plot(ibl.dist_of_endstates(sim_endstates))
        axes[x,y].set_ylim(0,100)
        counter = counter+1
        print(x,' ',y,' ',counter)

axes[0,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_xlabel('System State', labelpad=10,size=5)
axes[1,1].set_xlabel('System State', labelpad=10,size=5)
axes[1,2].set_xlabel('System State', labelpad=10,size=5)
axes[1,3].set_xlabel('System State', labelpad=10,size=5)

axes[1,0].tick_params(axis='x', labelsize=5)
axes[1,1].tick_params(axis='x', labelsize=5)
axes[1,2].tick_params(axis='x', labelsize=5)
axes[1,3].tick_params(axis='x', labelsize=5)
axes[0,0].tick_params(axis='y', labelsize=7)
axes[1,0].tick_params(axis='y', labelsize=7)

plt.subplots_adjust(wspace=.2,hspace=.2,left=0.15,bottom=0.15,right=0.90,top=0.90,)
plt.savefig(f'{graph_out_dir}IBL_Gibbs_2X4.png',dpi=300,bbox_inches='tight')
#plt.show()

#CAN
noise_list = [20,8,6,2,1,.5,.125,.01]
fig, axes = plt.subplots(2,4,figsize=(3,5),sharex=True,sharey=True)

counter = 0

for x in range(0,2):
    for y in range(0,4):
        ns_fn = noise_list[counter]
        sim_name = f'can_beta_{ns_fn}'
        sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
        dist_endstates = ibl.dist_of_endstates(sim_endstates)
        axes[x,y].plot(ibl.dist_of_endstates(sim_endstates))
        axes[x,y].set_ylim(0,100)
        counter = counter+1
        print(x,' ',y,' ',counter)

axes[0,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_ylabel('Frequency', labelpad=10,size=10)
axes[1,0].set_xlabel('System State', labelpad=10,size=5)
axes[1,1].set_xlabel('System State', labelpad=10,size=5)
axes[1,2].set_xlabel('System State', labelpad=10,size=5)
axes[1,3].set_xlabel('System State', labelpad=10,size=5)

axes[1,0].tick_params(axis='x', labelsize=5)
axes[1,1].tick_params(axis='x', labelsize=5)
axes[1,2].tick_params(axis='x', labelsize=5)
axes[1,3].tick_params(axis='x', labelsize=5)
axes[0,0].tick_params(axis='y', labelsize=7)
axes[1,0].tick_params(axis='y', labelsize=7)

plt.subplots_adjust(wspace=.2,hspace=.2,left=0.15,bottom=0.15,right=0.90,top=0.90,)
plt.savefig(f'{graph_out_dir}CAN_Gibbs_2X4.png',dpi=300,bbox_inches='tight')


'''COMBINE INTO ONE PLOT'''
plt.style.use('fivethirtyeight')
fig, axes = plt.subplots(2,4,figsize=(5,5),sharex=True,sharey=True)
noise_list = [.2,.5,.9,6]
panel_list = ['A','B','C','D']
counter = 0
for x in range(0,1):
    for y in range(0,4):
        ns_fn = noise_list[counter]
        sim_name = f'ibl_noise_{ns_fn}'
        sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
        dist_endstates = ibl.dist_of_endstates(sim_endstates)
        axes[x,y].plot(ibl.dist_of_endstates(sim_endstates))
        axes[x,y].set_ylim(0,100)
        axes[x,y].text(20,85,f'Noise={ns_fn}',fontsize=8)
        axes[x,y].text(850,95,panel_list[counter],fontsize=8)
        counter = counter+1
        #print(x,' ',y,' ',counter)

axes[0,0].set_ylabel('Frequency (ACT-R)', labelpad=10,size=10)
axes[1,0].set_ylabel('Frequency (CAN)', labelpad=10,size=10)
axes[1,0].set_xlabel('Retrieval Patt.', labelpad=10,size=9)
axes[1,1].set_xlabel('Retrieval Patt.', labelpad=10,size=9)
axes[1,2].set_xlabel('Retrieval Patt.', labelpad=10,size=9)
axes[1,3].set_xlabel('Retrieval Patt.', labelpad=10,size=9)

axes[1,0].tick_params(axis='x', labelsize=8)
axes[1,1].tick_params(axis='x', labelsize=8)
axes[1,2].tick_params(axis='x', labelsize=8)
axes[1,3].tick_params(axis='x', labelsize=8)
axes[0,0].tick_params(axis='y', labelsize=8)
axes[1,0].tick_params(axis='y', labelsize=8)
axes[1,0].set_xticks([10,1013])
axes[1,0].set_xticklabels(labels=['10','1013'])

#CAN
noise_list = [9,2,1,.125]
noise_list_invert = [.11,.5,1,8]
panel_list = ['A*','B*','C*','D*']
counter = 0

for x in range(1,2):
    for y in range(0,4):
        ns_fn = noise_list[counter]
        sim_name = f'can_beta_{ns_fn}'
        sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
        dist_endstates = ibl.dist_of_endstates(sim_endstates)
        axes[x,y].plot(ibl.dist_of_endstates(sim_endstates))
        axes[x,y].set_ylim(0,100)
        axes[x,y].text(20,85,f'Noise={noise_list_invert[counter]}',fontsize=8)
        axes[x,y].text(850,95,panel_list[counter],fontsize=8)
        counter = counter+1
        #print(x,' ',y,' ',counter)

plt.subplots_adjust(wspace=.2,hspace=.2,left=0.15,bottom=0.15,right=0.90,top=0.90,)
plt.savefig(f'{graph_out_dir}CAN_Gibbs_2X4.png',dpi=300,bbox_inches='tight')
#plt.show()
#EOF