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

'''
NOTES: 
BE CAREFUL WHEN RUNNING THE SIM RUNS LOOPS
--Not want to run them over if playing with the ibl can comparison.
--for ibl can comparison, see last code block which only reads in 
csvs from sims in sim blocks.
'''

'''ORIGINAL DATA AS COMPARISON'''
real_sim_endstates = ibl.extract_endstates(f'{data_in_dir}Obama.csv')
real_dist_endstates = ibl.dist_of_endstates(real_sim_endstates)
plt.style.use('fivethirtyeight')
fig, axes = plt.subplots(1,1,figsize=(5,5),sharex=True,sharey=True)
axes.plot(ibl.dist_of_endstates(real_sim_endstates))
axes.set_ylabel('Frequency', labelpad=3,size=10)
axes.set_xlabel('Retrieval Dec. Index', labelpad=3,size=10)
axes.tick_params(axis='x', labelsize=5)
axes.tick_params(axis='y', labelsize=7)

#axes.set_xticks([10,1013])
#axes.set_xticklabels(labels=['10','1013'],rotation=45)
axes.set_xticks([0,10,100,500,1000,1023])
axes.set_xticklabels(labels=[0,10,100,500,1000,1023],rotation=60)
plt.savefig(f'{graph_out_dir}Fig_Gibbs_RealData.png',dpi=400,bbox_inches='tight')

'''COMBINE INTO ONE PLOT'''



axes[0,0].set_ylabel('Frequency (IBL)', labelpad=10,size=10)
axes[1,0].set_ylabel('Frequency (CAN)', labelpad=10,size=10)
axes[1,0].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,1].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,2].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)
axes[1,3].set_xlabel('Retrieval Dec. Index', labelpad=10,size=6)

axes[1,0].tick_params(axis='x', labelsize=5)
axes[1,1].tick_params(axis='x', labelsize=5)
axes[1,2].tick_params(axis='x', labelsize=5)
axes[1,3].tick_params(axis='x', labelsize=5)
axes[0,0].tick_params(axis='y', labelsize=7)
axes[1,0].tick_params(axis='y', labelsize=7)
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
        axes[x,y].text(20,85,f'Noise = {noise_list_invert[counter]}',fontsize=5)
        axes[x,y].text(850,95,panel_list[counter],fontsize=7)
        counter = counter+1
        #print(x,' ',y,' ',counter)

plt.subplots_adjust(wspace=.2,hspace=.2,left=0.15,bottom=0.15,right=0.90,top=0.90,)
plt.savefig(f'{graph_out_dir}CAN_Gibbs_2X4.png',dpi=300,bbox_inches='tight')
#plt.show()
#EOF





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


'''PLOTS'''
#COMPARISION PLOT QUICK RUN AFTER SIMS
plt.plot(can_num_att_by_noise)
plt.plot(ibl_num_att_by_noise)
#plt.savefig(f'{graph_out_dir}Num_Endstates_CAN-IBL.png',dpi=300,bbox_inches='tight')
plt.show()


#USABLE FOR PUBLICATION
#IBL
noise_list = [.01,.1,.20,.30,.40,.5,.6,.7,.8,.9,1,1.5,2,4,6,8,10]
catch_num_attractors = np.array([])
for i in noise_list:
    #INITS
    ns_fn = i
    sim_name = f'ibl_noise_{ns_fn}'
    #BEGIN ANALYSIS
    sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    catch_num_attractors = np.append(catch_num_attractors,len(sim_endstates[0]))

#WRITE FOR COMPARE TO ISING SAMPLER SIMs   
ibl_num_att_by_noise = catch_num_attractors

#CAN - ISING SAMPLER SIMS
#NOTE CAN SIMS RUN ALREADY IN R
noise_list = [20,10,9,8,7,6,5,4,3,2,1,.5,.25,.125,.075,.03,.01]
catch_num_attractors = np.array([])
for i in noise_list:
    #INITS
    ns_fn = i
    sim_name = f'can_beta_{ns_fn}'
    #BEGIN ANALYSIS
    sim_endstates = ibl.extract_endstates(f'{data_out_dir}{sim_name}.csv')
    catch_num_attractors = np.append(catch_num_attractors,len(sim_endstates[0]))
    
can_num_att_by_noise = catch_num_attractors



#COMPARISION PLOT
x = np.arange(1,18)
fig, axes = plt.subplots(1,1,figsize=(5,5))

axes.plot(x,can_num_att_by_noise,color='blue',marker='o',dashes=[0,2,2,2],label='CAN',linewidth=1)
axes.plot(x,ibl_num_att_by_noise,color='black',marker='+',dashes=[0,0,2,2],label='IBL',linewidth=1)
axes.legend()

axes.set_xlabel('Noise Level (IBL)', labelpad=2,size=10)
axes.set_ylabel('Number of Unique Endstates', labelpad=1,size=10)

axes.tick_params(axis='x', labelsize=7)
axes.tick_params(axis='y', labelsize=7)
axes.set_xticks(x)
axes.set_xticklabels(labels=['.01','.1','.20','.30','.40','.5','.6','.7','.8','.9','1','1.5','2','4','6','8','10'])
#SECOND AXIS
axes2 = axes.twiny()
axes2.set_xlabel('Noise Level (CAN)', labelpad=2,size=10)
axes2.tick_params(axis='x', labelsize=7)
axes2.set_xticks(x)
axes2.set_xticklabels(labels=['0.05','0.1','.11','.125','.14','.17','.2','.25','.33','.5','1','2','4','8','13','33','100'])
plt.savefig(f'{graph_out_dir}Num_Endstates_CAN-IBL.png',dpi=300,bbox_inches='tight')
#plt.show()

#EOF