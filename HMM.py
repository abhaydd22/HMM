#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries to perform HMM
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hmmlearn as hmm
from hmmlearn.hmm import GaussianHMM


# In[2]:


# Get quotes from csv file for Nifty index 15 minutes data for last 10 years.
raw_data = pd.read_csv(r"F:\Abhay_New\Abhay\Projects\HMM\Nifty_15Mins_IEOD_10 Years.csv")


# In[3]:


raw_data.head()


# In[4]:


# Extract required details for modelling.
dates = np.array(raw_data['Date/Time'])
close_price = np.array(raw_data['Close'])
volume = np.array(raw_data['Volume'])


# In[5]:


# Take difference of closing price and compute rate of change
diff_percetage = 100.0*np.diff(close_price)/close_price[:-1]
dates =dates[1:]
volume = volume[1:]


# In[6]:


diff_percetage[0:5]


# In[7]:


dates[0:5]


#     

# In[8]:


volume[0:5]


# In[9]:


# Stack percetage difference and volume for columnwise for training.
X = np.column_stack([diff_percetage,volume])


# In[10]:


# create and train Gaussian HMM
print("\nTraining HMM...")
model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
model.fit(X)


# In[90]:


# Generate data using model
num_samples = 25
samples,_=model.sample(num_samples)
plt.plot(np.arange(num_samples),samples[:,0], c='blue')
plt.show()


# In[91]:


np.arange(num_samples)


# In[92]:


samples[:,0]


# 

# In[32]:


samples[:,1]


# In[94]:


# Predict the hidden states of HMM 
hidden_states = model.predict(X)


# In[95]:


hidden_states


# In[97]:


print("\nMeans and variances of hidden states:")
for i in range(model.n_components):
    print("\nHidden state", i+1)
    print("Mean =", round(model.means_[i][0], 3))
    print("Variance =", round(np.diag(model.covars_[i])[0], 3))


# In[ ]:




