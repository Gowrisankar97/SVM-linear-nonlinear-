#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importig libary
import pandas as pd


# In[2]:


df=pd.read_csv("iris.csv")
df.tail()


# In[3]:


#seprating target varibale  from dataset(iris dataset)
X=df.iloc[:,1:5]

Y=df["Species"]
Y.value_counts()


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


xtrain,xtest,ytrain,ytest=train_test_split(X,Y,train_size=0.7,random_state=1)


# In[6]:


#import svm from sklearn
from sklearn.svm import SVC


# In[7]:


svm=SVC(kernel="linear",C=1)


# In[8]:


svm.fit(xtrain,ytrain)


# In[9]:


print(svm.score(xtrain,ytrain))
from mlxtend.plotting import plot_decision_regions


# In[ ]:





# In[10]:


#accuracy is 98% percent if value of c changes the accuracy also will change


# In[11]:


#kernel method


# In[12]:


import matplotlib.pyplot as plt
import numpy as np


# In[13]:


np.random.seed(1)


# In[14]:


x_xor=np.random.randn(200,2)


# In[15]:


y_xor=np.logical_xor(x_xor[:, 0] > 0,x_xor[:, 1] > 0)


# In[16]:


y_xor=np.where(y_xor,1,-1)


# In[17]:


from mlxtend.plotting import plot_decision_regions
plt.scatter(x_xor[y_xor == 1, 0],x_xor[y_xor == 1, 1],c='b', marker='x',label='1')
plt.scatter(x_xor[y_xor == -1, 0], x_xor[y_xor == -1, 1], c='r', marker='s',label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()


# In[18]:


#introducing rbf function 
svm=SVC(kernel="rbf",C=100,gamma=0.5)


# In[19]:


svm.fit(x_xor,y_xor)


# In[20]:


plot_decision_regions(x_xor,y_xor,clf=svm)
#By changing the  value of gamma the decision surface will change


# In[ ]:




