import pyactup
import numpy as np

m = pyactup.Memory()
m.learn({'ang':1,'kind':1,'hon':1})
m.learn({'ang':1,'kind':1,'hon':-1})
m.learn({'ang':1,'kind':-1,'hon':1})
m.learn({'ang':1,'kind':-1,'hon':-1})
m.learn({'ang':-1,'kind':1,'hon':1})
m.learn({'ang':-1,'kind':1,'hon':-1})
m.learn({'ang':-1,'kind':-1,'hon':1})
m.learn({'ang':-1,'kind':-1,'hon':-1})
m.advance()

m.advance()

m.retrieve({'ang':1})

m.blend('ang',{'kind':-1,'hon':-1})
    



