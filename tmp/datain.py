import numpy as np
import pandas as pd
import pyactup

df = pd.read_csv('Obama.csv')

'''BEGIN FUNCS'''

def load_m(m,df):
    '''
    m is existing pyactup memory object
    df is the binary data, e.g. us election 
    survey data from ANES.
    '''
#ENCODE DATA
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

#TEST FUNCS
#A FULL SIM
ns = 10
num_retrieve = 100
sim_name = f'ibl_noise_{ns}_0'
m = pyactup.Memory(noise=ns)
load_m(m,df)
m.advance()
sim = sim_no_cue(m,num_retrieve)#the numpy sim output array
df_sim = pd.DataFrame(sim,columns=df.columns)
df_sim.to_csv(f'{sim_name}.csv',index=None)

'''END FUNCS'''

'''MEMORY VERY HIGH NOISE'''
m = pyactup.Memory(noise=10)

#ENCODE DATA
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

m.advance()

#SIMULATE RETRIEVAL
catch = np.array([9,9,9,9,9,9,9,9,9,9])
for i in range(0,100):
    tmp = m.retrieve()
    tmp2 = np.array(list(tmp.values()))
    catch = np.vstack((catch,tmp2))
use = catch[1:]

#WRITE SIM TO FILE
df_use = pd.DataFrame(use,columns=df.columns)
df_use.to_csv('ibl_noise_1000.csv',index=None)


'''MEMORY HIGH NOISE'''
m = pyactup.Memory(noise=.95)

#ENCODE DATA
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

m.advance()

#SIMULATE RETRIEVAL
catch = np.array([9,9,9,9,9,9,9,9,9,9])
for i in range(0,100):
    tmp = m.retrieve()
    tmp2 = np.array(list(tmp.values()))
    catch = np.vstack((catch,tmp2))
use = catch[1:]

#WRITE SIM TO FILE
df_use = pd.DataFrame(use,columns=df.columns)
df_use.to_csv('ibl_noise_095.csv',index=None)



'''MEMORY NOISE'''
m = pyactup.Memory(noise=.50)

#ENCODE DATA
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

m.advance()

#SIMULATE RETRIEVAL
catch = np.array([9,9,9,9,9,9,9,9,9,9])
for i in range(0,100):
    tmp = m.retrieve()
    tmp2 = np.array(list(tmp.values()))
    catch = np.vstack((catch,tmp2))
use = catch[1:]

#WRITE SIM TO FILE
df_use = pd.DataFrame(use,columns=df.columns)
df_use.to_csv('ibl_noise_050.csv',index=None)


'''MEMORY LOW NOISE'''
m = pyactup.Memory(noise=.05)

#ENCODE DATA
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

m.advance()

#SIMULATE RETRIEVAL
catch = np.array([9,9,9,9,9,9,9,9,9,9])
for i in range(0,100):
    tmp = m.retrieve()
    tmp2 = np.array(list(tmp.values()))
    catch = np.vstack((catch,tmp2))
use = catch[1:]

#WRITE SIM TO FILE
df_use = pd.DataFrame(use,columns=df.columns)
df_use.to_csv('ibl_noise_005.csv',index=None)



'''MEMORY .25 WITH SELECT RETRIVAL ON HNS'''
m = pyactup.Memory(noise=.25)

#ENCODE DATA
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

m.advance()

#SIMULATE RETRIEVAL
catch = np.array([9,9,9,9,9,9,9,9,9,9])
for i in range(0,100):
    tmp = m.retrieve()
    tmp2 = np.array(list(tmp.values()))
    catch = np.vstack((catch,tmp2))
use = catch[1:]

#WRITE SIM TO FILE
df_use = pd.DataFrame(use,columns=df.columns)
df_use.to_csv('ibl_noise_025_cueNone.csv',index=None)


'''MEMORY .25 WITH SELECT RETRIVAL ON HNS'''
m = pyactup.Memory(noise=.25)

#ENCODE DATA
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

m.advance()

#SIMULATE RETRIEVAL
catch = np.array([9,9,9,9,9,9,9,9,9,9])
for i in range(0,100):
    tmp = m.retrieve({'Hns':1})
    tmp2 = np.array(list(tmp.values()))
    catch = np.vstack((catch,tmp2))
use = catch[1:]

#WRITE SIM TO FILE
df_use = pd.DataFrame(use,columns=df.columns)
df_use.to_csv('ibl_noise_025_cueHNS.csv',index=None)

'''TESTING '''
#BUILD SIM
