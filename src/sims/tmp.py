

#TEST CUED SIM.
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

#TESTING CODE
m = pyactup.Memory(noise=0.5)
ibl.load_m(m,df)
m.advance()
m.chunks
{'Mor': 1, 'Led': 1, 'Car': 0, 'Kno': 1, 'Int': 0, 'Hns': 0, 'Ang': 0, 'Hop': 0, 'Afr': 1, 'Prd': 0}

m.retrieve({'Mor':1})

sim_w_cue(m,10,'Mor')

#CLOSE TO RIGHT
cue_list = ['Mor','Led','Car', 'Kno', 'Int', 'Hns', 'Ang', 'Hop', 'Afr', 'Prd']
for i in cue_list:
    sim_w_cue(m,10,i)