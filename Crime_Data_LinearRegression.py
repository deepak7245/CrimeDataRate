#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


data = pd.read_csv("crime_data.csv")


# In[23]:


data = data.rename(columns={data.columns[0]: 'Cities'})


# In[24]:


data.head()


# In[25]:


data.info()


# In[26]:


# Summarizing the dataset
data.describe()


# In[27]:


# Cheack the missing values
data.isnull().sum()


# In[28]:


# Exploratory Data Analysis
# Correlation
data.corr()


# In[29]:


# Visualization the data by paiplots
import seaborn as sns
sns.pairplot(data)


# In[30]:


# Analyzing the correlation data
data.corr()


# In[31]:


plt.scatter(data['Rape'], data['UrbanPop'])
plt.xlabel('Rape')
plt.ylabel('UrbanPop')


# In[32]:


import seaborn as sns
sns.regplot(x="Rape",y="UrbanPop",data=data)


# In[33]:


plt.scatter(data['Assault'], data['UrbanPop'])
plt.xlabel("Assault")
plt.ylabel("UrbanPop")


# In[34]:


plt.scatter(data['Cities'], data['Rape'])
plt.xlabel("Cities")
plt.ylabel("Rape")


# In[37]:


# Plot this residuals 

sns.displot(data,kind="kde")


# In[38]:


sns.displot(data)


# In[55]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

data["Cities"] = LE.fit_transform(data["Cities"])


# In[56]:


# Data partition X and Y
# Dependent and Independent Feature
x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[57]:


data.head()


# In[58]:


x.head()


# In[59]:


y.head()


# In[60]:


# split the data train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=30)


# In[61]:


x_train


# In[62]:


x_test


# In[63]:


## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[64]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[65]:


import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))


# In[66]:


x_train


# In[67]:


x_test


# # Model Training

# In[69]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

## print the coefficients and the intercept
print(regression.coef_)


# In[70]:


print(regression.intercept_)


# In[71]:


## on which parameters the model has been trained
regression.get_params()


# In[73]:


### Prediction With Test Data
reg_pred=regression.predict(x_test)


# In[74]:


reg_pred


# # Assumptions

# In[75]:


## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)


# In[76]:


## Residuals
residuals=y_test-reg_pred


# In[77]:


residuals


# In[85]:


## Plot this residuals 

sns.displot(residuals,kind="kde")


# In[79]:


## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(reg_pred,residuals)


# In[80]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# # R-Square and Adjust R-Square
#  
#  Formula
#     
#     R^2 = 1-SSR/SST

# In[81]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# # Adjust R^2
# R^2 = 1-[(1-R^2)*(n-1)/(n-k-1)]

# In[83]:


#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# # pickling The Data File Deployment

# In[88]:


import pickle


# In[89]:


pickle.dump(regression,open('regmodel.pkl','wb'))


# In[90]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[ ]:




