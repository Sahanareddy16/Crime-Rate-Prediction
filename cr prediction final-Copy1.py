#!/usr/bin/env python
# coding: utf-8

# In[357]:


# importing modules and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing


# In[358]:


# importing data
crimes = pd.read_csv(r"C:\Users\sahan\OneDrive\Documents\PDF X\01_District_wise_crimes_committed_IPC_2001_2012.csv")


# In[359]:


crimes


# In[360]:


crimes.columns


# In[361]:


df1=crimes.drop(['DISTRICT','ATTEMPT TO MURDER', 'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER','CUSTODIAL RAPE',
       'OTHER RAPE', 'KIDNAPPING & ABDUCTION',
       'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
       'KIDNAPPING AND ABDUCTION OF OTHERS', 'DACOITY',
       'PREPARATION AND ASSEMBLY FOR DACOITY', 'BURGLARY',
       'AUTO THEFT', 'OTHER THEFT', 'RIOTS', 'CRIMINAL BREACH OF TRUST',
       'CHEATING', 'COUNTERFIETING', 'ARSON', 'HURT/GREVIOUS HURT', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
       'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
       'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
       'CAUSING DEATH BY NEGLIGENCE', 'OTHER IPC CRIMES'],axis=1)


# In[362]:


df1.columns


# In[363]:


df1.isna().all()


# In[364]:


df=df1.rename(columns={'TOTAL_IPC_CRIMES':'IPC_CRIMES'})


# In[365]:


label_encoder = preprocessing.LabelEncoder()
df['STATE'] = label_encoder.fit_transform(df['STATE'])


# In[366]:


df


# In[375]:


df.rename(columns={'TOTAL_IPC_CRIMES':'IPC_CRIMES'})


# In[376]:


df


# In[377]:


df.head()


# In[378]:


df.tail()


# In[379]:


df.info()


# In[380]:


df.describe()


# In[381]:


corr = df.corr()
plt.subplots(figsize=(25,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[382]:


# SCatter plot
plt.scatter(df.STATE,df.MURDER)
plt.title("Distribution of datapoints")
plt.show()


# In[383]:


sns.set_context("paper")
sns.scatterplot(x='STATE', y='MURDER', data=df, hue='YEAR')


# In[384]:


sns.set_context("paper")
sns.scatterplot(x='IPC_CRIMES', y='MURDER', data=df, hue='STATE')


# In[385]:


# Generate relation ship plot between Age and Estimated Salary based on Gender
sns.set_context("poster")
sns.relplot(data=df, x="THEFT", y="ROBBERY", hue="STATE")


# In[386]:


sns.set_context("poster")
sns.relplot(data=df, x="DOWRY_DEATHS", y="IPC_CRIMES", hue="STATE",
 col="YEAR",col_wrap=3)


# In[387]:


# PairPlot
sns.set_style("ticks")
sns.pairplot(df,hue = 'THEFT',diag_kind = "kde",kind = "scatter",
             palette = "husl")
plt.show()


# In[388]:


sns.pairplot(df)


# In[389]:


extraval=df
extraval.head()


# In[390]:


sns.pointplot(x="STATE",y="MURDER",data=extraval)
plt.show()


# In[391]:


sns.jointplot(x='STATE',y='MURDER',data=df, kind='reg')


# In[392]:


ax = sns.boxplot(x=df['STATE'])


# In[393]:


ax = sns.violinplot(x=df['MURDER'])


# In[394]:


ax = sns.violinplot(x=df['STATE'])


# In[395]:


ax = sns.regplot(x='IPC_CRIMES', y='MURDER', data=df,ci=None)


# In[396]:


df.hist(alpha=1.0, figsize=(20,15))


# In[397]:


correlation=df.corr()
df.corr()


# In[398]:


plt.figure(figsize=(10,8))
sns.barplot(x='YEAR',y='IPC_CRIMES',data=df)


# In[400]:


# Assigning feature and target variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[401]:


#Setting the value for X and Y
X = df[['YEAR','MURDER', 'RAPE','THEFT','ROBBERY','IPC_CRIMES']]
y = df['STATE']


# In[402]:


#Splitting the dataset
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.26, random_state = 100)


# In[407]:


mlr = LinearRegression()  
mlr.fit(X_train, y_train)


# In[408]:


#Intercept and Coefficient
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(X, mlr.coef_))


# In[409]:


#Prediction of test set
y_pred_mlr= mlr.predict(X_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))


# In[410]:


#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()


# In[411]:


#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(X,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[404]:





# In[406]:


#Future prediction

future_data = pd.DataFrame({
    'MURDER': [85,140,85,93],
    'THEFT': [351,547,1208,460],
     'RAPE':[67,22,33,27],
     'IPC_CRIMES':[5312,5181,7268,5510],
     'YEAR':[2009,2010,2011,2012],
    'ROBBERY':[20,24,17,9]})

# Select the input parameters for the prediction
future_x = future_data[['YEAR','MURDER', 'THEFT','RAPE','IPC_CRIMES','ROBBERY']]


# Use the model to predict the number of new cases for the future dates
future_data['df'] = mlr.predict(future_x)

# Print the predicted number of new cases for each future date
print(future_data[['df']])


# In[ ]:




