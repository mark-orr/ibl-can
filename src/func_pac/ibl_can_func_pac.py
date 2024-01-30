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

def sim_w_cue(m,n,x):
    '''
    m is pyactup memory instance
    n is the number of simulations to run
    x is a str of the chunk key for cue
    this returns a two dim numpy array, each inner
    is a retrieval
    '''
    #SIMULATE RETRIEVAL
    catch = np.array([9,9,9,9,9,9,9,9,9,9])
    for i in range(0,n):
        tmp = m.retrieve({x:1})
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


def wass(b,t):
    '''
    b is base distribution (expects np.array of integers)
    t is test distribution (expects same)
    returns wass number not normalized
    note: if want to normalize it, divid wass by len of dist minuse
    1 as this is dividing by the max
    '''
    b_prob = b/b.sum()
    t_prob = t/t.sum()
    
    return np.sum(np.abs(np.cumsum(b_prob)-np.cumsum(t_prob)))

def wass_norm(b,t):
    '''
    b is base distribution (expects np.array of integers)
    t is test distribution (expects same)
    returns wass number not normalized
    note: if want to normalize it, divid wass by len of dist minuse
    1 as this is dividing by the max
    '''
    b_prob = b/b.sum()
    t_prob = t/t.sum()

    wass_raw = np.sum(np.abs(np.cumsum(b_prob)-np.cumsum(t_prob)))
   
    return wass_raw/len(b_prob)

#TEST
#a = np.array([10,0,0,0,0,0,0,0,0,0])
#b = np.array([0,0,0,0,0,0,0,0,0,10])
#wass(a,b)
#EOF