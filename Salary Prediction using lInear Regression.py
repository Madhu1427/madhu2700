#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[106]:


data = pd.read_csv("C://Users//ADMIN//Desktop//madhu//Salary_Data.csv")


# In[107]:


data.head()


# In[108]:


#check if there is anynull values
data.isnull()


# In[109]:


data.describe()


# In[110]:


x =data.iloc[:,0]
y = data.iloc[:,1]


# In[111]:


x.ndim


# In[112]:


x=x.values.reshape(-1,1)


# In[113]:


y=y.values.reshape(-1,1)


# In[114]:


#visulization of data
plt.plot(x,y,marker="o")
plt.title("Salary Based Experience")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show


# In[115]:


#import sklearn using train_test_split to split data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state =2)


# In[116]:


x_test


# In[117]:


y_test


# In[118]:


linearreg = LinearRegression()
linearreg.fit(x_train,y_train)
y_pred = linearreg.predict(x_test)


# In[119]:


x_test


# In[120]:


y_pred


# In[121]:


#to check the accuracy
from sklearn.metrics import mean_squared_error as mse
MSE = mse(y_test,y_pred)
MSE


# In[122]:


#PLOT ACTUAL VS PREDICTED
plt.scatter(x_test,y_test)
plt.scatter(x_test,y_pred)
plt.xlabel("years of expeience")
plt.ylabel("Salary")
plt.show()


# In[123]:


linearreg.predict([[2.7]])


# In[124]:


linearreg.intercept_


# In[126]:


linearreg.coef_

