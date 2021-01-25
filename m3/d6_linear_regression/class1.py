#!/usr/bin/env python
# coding: utf-8

# In[87]:


class LinearRegression():
    def __init__(self,a=0,b=0):
        self.a=a
        self.b=b

    def loading(self,file):
        import pandas as pd
        self.data=pd.read_csv(f"{file}",header=None)
        return self.data
            
    def covar(self,x,y):
        import numpy as np
        res=0
        for i in range(len(x)): 
            res+=(x[i]-np.mean(x))*(y[i]-np.mean(y))
        return res
    
    def var(self,x):
        res2=0
        import numpy as np
        for i in range(len(x)):
            res2+=(x[i]-np.mean(x))**2
        return res2 
    
    def fit(self,x,y):
        import numpy as np
        self.b+=self.covar(x,y)/self.var(x)
        self.a+=np.mean(y)-(self.b*np.mean(x))
        return self
    
    def plot1(self,x,y):
        import matplotlib.pyplot as plt
        plt.scatter(x,y)
    
    def plot2(self,x):
        import matplotlib.pyplot as plt
        plt.plot(x,self.a+self.b*x)
    
    def plot3(self,x,y):
        import matplotlib.pyplot as plt
        plt.scatter(x,y)
        plt.plot(x,self.a+x*self.b,color='#ff0000')
        
    def predict(self,x,y):
        import numpy as np
        mse=sum([(y[i]-(self.a+self.b*x[i]))**2 for i in range(len(x))])
        rmse=np.sqrt(mse)
        score=1-(mse/self.var(y))
        return score

