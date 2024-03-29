'''FUNCTIONS PAC'''
import numpy as np
import pandas as pd
import pyactup
import matplotlib.pyplot as plt

def load_m(m,df):
    '''
    m is existing pyactup memory object
    df is the binary data, e.g. us election 
    survey data from ANES.
    '''
    l = len(df)
    for i in range(0,l):
        m.learn({df.columns[0]:df.iloc[i][df.columns[0]],
                df.columns[1]:df.iloc[i][df.columns[1]],           
                df.columns[2]:df.iloc[i][df.columns[2]],
                df.columns[3]:df.iloc[i][df.columns[3]],
                df.columns[4]:df.iloc[i][df.columns[4]],
                df.columns[5]:df.iloc[i][df.columns[5]],
                df.columns[6]:df.iloc[i][df.columns[6]],
                df.columns[7]:df.iloc[i][df.columns[7]],
                df.columns[8]:df.iloc[i][df.columns[8]],
                df.columns[9]:df.iloc[i][df.columns[9]]})
    
    return m

def sim_no_cue(m,n):
    '''
    m is pyactup memory instance
    n is the number of simulations to run
    this returns a two dim numpy array, each inner
    is a retrieval
    '''
    #SIMULATE RETRIEVAL
    catch = np.array([9,9,9,9,9,9,9,9,9,9])
    for i in range(0,n):
        tmp = m.retrieve()
        tmp2 = np.array(list(tmp.values()))
        catch = np.vstack((catch,tmp2))
    use = catch[1:]

    return use

def convert_dec_to_array(x):
    mystr = format(x, '010b') #WITH PADDING
    return np.fromiter(mystr, dtype=int)

def extract_endstates(d):
    '''
    d is data as csv is output of sim either r ising sampler
    or ibl-can pyactup eg ibl_noise_10_0.csv
    returns two-tuple np.array where 
    [0] is decimal version of each endstate
    [1] is the mapped frequency of [0]
    eg output was tmp_e_uniq_counts
    '''
    df = pd.read_csv(d)
    catch_endstate = np.array(df)
    catch_each_decimal = np.array([])
    for i in range(1,len(catch_endstate)):
        tmp0 = catch_endstate[i].astype(int)
        tmp1 = list(tmp0.astype(str))
        tmp2 = int(''.join(tmp1),2)
        catch_each_decimal = np.append(catch_each_decimal,tmp2)
        
    return np.unique(catch_each_decimal,return_counts=True)
#EXAMPLE USE   
#extract_endstates('ibl_noise_1000.csv')

def extract_endstates_for_plot(x):
    '''
    x is the frequency tuple (two) of two arrays.
    [0] is the list of decimal versions of the system states
    [1] is the mapped frequency of each system state to [0]
    x should be tmp_e_uniq_counts
    '''
    uniq_endstates_plot = np.full(10,99)
    for i in range(0,len(x[0])):
        x_1 = x[0][i].astype(int)
        x_2 = convert_dec_to_array(x_1)
        uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))
    
    return uniq_endstates_plot

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


#EXAMPLE USE


df = pd.read_csv('Obama.csv')

#TEST FUNCS
#A FULL SIM
ns = .85
ns_fn = '0_85'
num_retrieve = 100
sim_name = f'ibl_noise_{ns_fn}'
m = pyactup.Memory(noise=ns)
load_m(m,df)
m.advance()
sim = sim_no_cue(m,num_retrieve)#the numpy sim output array
df_sim = pd.DataFrame(sim,columns=df.columns)
df_sim.to_csv(f'{sim_name}.csv',index=None)

sim_endstates = extract_endstates(f'{sim_name}.csv')

plt.style.use('fivethirtyeight')
imgplot = plt.imshow(extract_endstates_for_plot(sim_endstates)[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
plt.savefig(f'Fig_Endstates_{sim_name}.png',dpi=400,bbox_inches='tight')
plt.show()
plt.clf()

plt.plot(dist_of_endstates(test1))
plt.savefig(f'Fig_Gibbs_{sim_name}.png',dpi=400,bbox_inches='tight')
plt.show()
plt.clf()

###TEST LOOP OVER NOISE

noise_list = [0,.25,.35,.40,.5,.6,.7,.8,.9,1,2,4,8,16]
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
print('Sim END')

#FOR R, does the extraction work?
#YES
sim_endstates = extract_endstates('can_beta_10.csv')
plt.style.use('fivethirtyeight')
imgplot = plt.imshow(extract_endstates_for_plot(sim_endstates)[1:],cmap="binary",aspect='auto',interpolation='none') 
plt.xlabel('State Vector Index')
plt.ylabel('Point-Attractor Index')
plt.colorbar()
#plt.savefig(f'Fig_Endstates_{sim_name}.png',dpi=400,bbox_inches='tight')
plt.show()
plt.clf()

plt.plot(dist_of_endstates(sim_endstates))
#plt.savefig(f'Fig_Gibbs_{sim_name}.png',dpi=400,bbox_inches='tight')
plt.show()
plt.clf()






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