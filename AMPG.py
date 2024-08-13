#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


autompg = pd.read_csv(r"C:\Users\Rahul\OneDrive\Desktop\autompg.csv")


# In[35]:


autompg


# In[36]:


autompg.shape


# In[37]:


autompg.describe()


# In[38]:


autompg.sample(5)


# In[39]:


autompg.head(5)


# In[40]:


autompg.tail(5)


# In[41]:


autompg.isnull().sum()


# In[42]:


ampg = autompg
ampg


# # Data Cleansing
# 

# In[43]:


ampg.drop('car name',axis=1,inplace=True)


# In[44]:


ampg.info()


# In[45]:


ampg.shape


# In[46]:


ampg['origin'].value_counts()


# In[47]:


ampg.nunique()


# # Preprocessing
# 

# In[48]:


ampg.isnull().sum()


# # Split the dataset

# In[49]:


X = ampg[['mpg']].values.reshape(398,1)
X.shape


# In[50]:


Y =  ampg[['cylinders','displacement','horsepower','weight','acceleration','model year', 'origin']].values.reshape(398, 7)
Y.shape


# # EDA

# In[56]:


# Step 1: Replace '?' with NaN
ampg.replace('?', np.nan, inplace=True)

# Step 2: Convert columns to numeric, coercing errors to NaN
ampg = ampg.apply(pd.to_numeric, errors='coerce')

# Correlation
f,ax = plt.subplots(figsize = [15,8])
sns.heatmap(ampg.corr(),annot=True, fmt=".2f",ax=ax,cmap="magma")
ax.set_title("Correlation Matrix", fontsize = 20)
plt.show()


# In[76]:


sns.pairplot(ampg,diag_kind='kde',markers='+')


# In[77]:


sns.barplot(x='origin',y='mpg',data=ampg)
# here 1--> USA, 2--> JAPAN, 3--> EUROPE based on the dataset


# In[78]:


plt.figure(figsize=[15,8])
sns.barplot(x=ampg['model year']+1900,y=ampg['mpg'])
plt.title("Consumption Gallon by year")


# In[79]:


# skewness
# feature - dependant variable
sns.displot(ampg['mpg'])
plt.show()


# In[80]:


sns.distplot(ampg['mpg'], fit=norm)


# # Train and Test Split method
# 

# In[82]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=12)


# # Apply the model

# In[86]:


# Replace placeholders with NaN
ampg.replace('?', np.nan, inplace=True)

# Convert all columns to numeric, coercing errors to NaN
ampg = ampg.apply(pd.to_numeric, errors='coerce')

# Separate features and target variable
X = ampg.drop('mpg', axis=1)
y = ampg['mpg']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with the median of each column
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Handle any potential missing values in the target variable
y_train = np.where(np.isnan(y_train), np.nanmedian(y_train), y_train)

# Create and train the linear regression model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# # Coefficient

# In[88]:


print("Model coefficients:", model.coef_)


# # Intercept

# In[89]:


print("Model intercept:", model.intercept_)


# # Predicting using test data

# In[91]:


y_pred = model.predict(X_test)
y_pred


# In[92]:


# prediction with new value
print(model.predict([[8,350,165,3693,11.5,70,1]]))


# # R2 Score

# In[93]:


print("R2 Score",r2_score(y_test, y_pred))

