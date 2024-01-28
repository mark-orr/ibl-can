import pyactup
import numpy as np
import matplotlib.pyplot as plt

m = pyactup.Memory()
m.learn({'ang':1,'kind':1,'hon':1})
m.learn({'ang':1,'kind':1,'hon':-1})
m.learn({'ang':1,'kind':-1,'hon':1})
m.learn({'ang':1,'kind':-1,'hon':-1})
m.learn({'ang':-1,'kind':1,'hon':1})
m.learn({'ang':-1,'kind':1,'hon':-1})
m.learn({'ang':-1,'kind':-1,'hon':1})
m.learn({'ang':-1,'kind':-1,'hon':-1})
for i in range(0,2):
    m.learn({'ang':1,'kind':1})
m.advance()
m.activation_history = []
m.chunks
m.activation_history
catch_1 = []
for i in range(0,99):
    print(m.retrieve({'ang':1})['kind'])
    print(m.retrieve({'ang':1}))
    catch.append(m.retrieve({'ang':1})['kind'])
plt.hist(catch)

for i in range(0,2):
    m.learn({'ang':1,'kind':1})
    m.advance()

catch_2 = []
for i in range(0,99):
    print(m.retrieve({'ang':1})['kind'])
    print(m.retrieve({'ang':1}))
    catch.append(m.retrieve({'ang':1})['kind'])
plt.hist(catch)


m.blend('ang',{'kind':-1,'hon':-1})
    



