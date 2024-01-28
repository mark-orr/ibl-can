import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_dec_to_array(x):
    mystr = format(x, '010b') #WITH PADDING
    return np.fromiter(mystr, dtype=int)

#IBL NOISE 10
df_ibl_noise_1000 = pd.read_csv('ibl_noise_1000.csv')
catch_endstate = np.array(df_ibl_noise_1000)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig('Fig_Endstates_IBL_Noise_1000.png',dpi=400,bbox_inches='tight')
'''TEST'''
full_freq_array = np.full((1024,),np.NaN)
b = tmp_e_uniq_counts[0]
c = tmp_e_uniq_counts[1]

for i in range(0,1024):
    if np.isin(i,b): 
        #print('isin b') 
        #Assign c to array at i
        full_freq_array[i] = int(np.where(b==i)[0])
    else: 
        #print('notin b')
        #ASSIGN o to array at i
        full_freq_array[i] = int(0)
        
plt.plot(full_freq_array)
plt.savefig('Fig_Gibbs_IBL_Noise_1000.png',dpi=400,bbox_inches='tight')
'''END TEST'''


#IBL NOISE .95
df_ibl_noise_095 = pd.read_csv('ibl_noise_095.csv')
catch_endstate = np.array(df_ibl_noise_095)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig('Fig_Endstates_IBL_Noise_095.png',dpi=400,bbox_inches='tight')
'''TEST'''
full_freq_array = np.full((1024,),np.NaN)
b = tmp_e_uniq_counts[0]
c = tmp_e_uniq_counts[1]

for i in range(0,1024):
    if np.isin(i,b): 
        print('isin b') 
        #Assign c to array at i
        full_freq_array[i] = int(np.where(b==i)[0])
    else: 
        print('notin b')
        #ASSIGN o to array at i
        full_freq_array[i] = int(0)
        
plt.plot(full_freq_array)
'''END TEST'''



#IBL NOISE .05
df_ibl_noise_005 = pd.read_csv('ibl_noise_005.csv')
catch_endstate = np.array(df_ibl_noise_005)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig('Fig_Endstates_IBL_Noise_005.png',dpi=400,bbox_inches='tight')


#CAN BETA 10
df_can_beta_10 = pd.read_csv('can_beta_10.csv')
catch_endstate = np.array(df_can_beta_10)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig('Fig_Endstates_CAN_Beta_10.png',dpi=400,bbox_inches='tight')


#CAN BETA 005
df_can_beta_005 = pd.read_csv('can_beta_005.csv')
catch_endstate = np.array(df_can_beta_005)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
#plt.savefig('Fig_Endstates_CAN_Beta_005.png',dpi=400,bbox_inches='tight')

'''TEST'''
full_freq_array = np.full((1024,),np.NaN)
b = tmp_e_uniq_counts[0]
c = tmp_e_uniq_counts[1]

for i in range(0,1024):
    if np.isin(i,b): 
        #print('isin b') 
        #Assign c to array at i
        full_freq_array[i] = int(np.where(b==i)[0])
    else: 
        #print('notin b')
        #ASSIGN o to array at i
        full_freq_array[i] = int(0)
        
plt.plot(full_freq_array)
plt.savefig('Fig_Gibbs_CAN_Beta_005.png',dpi=400,bbox_inches='tight')
'''END TEST'''


#IBL NOISE 025 with no cue
df_ibl_noise_025 = pd.read_csv('ibl_noise_025_cueNone.csv')
catch_endstate = np.array(df_ibl_noise_025)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig('Fig_Endstates_IBL_Noise_025_cueNone.png',dpi=400,bbox_inches='tight')


#IBL NOISE 025 with no cue
df_ibl_noise_025 = pd.read_csv('ibl_noise_025_cueHns.csv')
catch_endstate = np.array(df_ibl_noise_025)
catch_endstate.shape

catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)
    
tmp_e_uniq_counts = np.unique(catch_each_decimal,return_counts=True)

uniq_catch_endstates = tmp_e_uniq_counts
uniq_endstates_plot = np.full(10,99)
for i in range(0,len(uniq_catch_endstates[0])):
    x_1 = uniq_catch_endstates[0][i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(uniq_endstates_plot[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig('Fig_Endstates_IBL_Noise_025_cueHns.png',dpi=400,bbox_inches='tight')

#TMP TEST OF DISTRIBUTION AND WASS.
#DIST FIRST
'''Note for this study, we will use low noise low temperature models only'''
full_freq_array = np.full((1024,),np.NaN)
b = tmp_e_uniq_counts[0]
c = tmp_e_uniq_counts[1]

for i in range(0,1024):
    if np.isin(i,b): 
        print('isin b') 
        #Assign c to array at i
        full_freq_array[i] = c[int(np.where(b==i)[0])]
    else: 
        print('notin b')
        #ASSIGN o to array at i
        full_freq_array[i] = int(0)
        
plt.plot(full_freq_array)

def dist_of_endstates(x):
    '''
    x is the frequency tuple (two) of two arrays.
    [0] is the list of decimal versions of the system states
    [1] is the mapped frequency of each system state to [0]
    x should be tmp_e_uniq_counts
    note: 1024 is 2 to the 10 vertices.
    note: can just plt.plot(d
    '''
    full_freq_array = np.full((1024,),np.NaN)
    b = x[0]
    c = x[1]
    for i in range(0,1024):
        if np.isin(i,b): 
            print('isin b') 
            #Assign c to array at i
            full_freq_array[i] = c[int(np.where(b==i)[0])]
        else: 
            print('notin b')
            #ASSIGN o to array at i
            full_freq_array[i] = int(0)
    return full_freq_array


#EOF
