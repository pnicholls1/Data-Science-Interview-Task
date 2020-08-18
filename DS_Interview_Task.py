#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# # Read data in and clean

# In[2]:


#read in data from local directory, treat ' ?' values as null

address = r'/Users/pete/TLM/Earnings.csv'

earnings_data = pd.read_csv(address, na_values=' ?')

earnings_data.head()


# In[3]:


# find dimensions of dataset
earnings_data.shape


# In[4]:


# find information of dataset - for nulls in particular
earnings_data.info()


# In[5]:


# find some basic descriptive stats about dataset
earnings_data.describe()


# In[6]:


#check how many null values in data
earnings_data.isnull().sum()


# In[7]:


#drop rows with null values
earnings_data = earnings_data.dropna(axis=0)


# In[8]:


#reset index after dropping rows
earnings_data = earnings_data.reset_index()


# In[9]:


# drop new index row created by reset_index command
earnings_data.drop('index', axis=1, inplace=True)


# In[10]:


#check nulls have been removed
earnings_data.isnull().sum()


# In[11]:


#trim whitespace from string fields
earnings_data['marital-status']=earnings_data['marital-status'].str.strip()
earnings_data['occupation']=earnings_data['occupation'].str.strip()
earnings_data['relationship']=earnings_data['relationship'].str.strip()
earnings_data['race']=earnings_data['race'].str.strip()
earnings_data['gender']=earnings_data['gender'].str.strip()
earnings_data['native-country']=earnings_data['native-country'].str.strip()
earnings_data['class']=earnings_data['class'].str.strip()


# In[12]:


#check predictant variable is binary as required for the classifier
sns.countplot(earnings_data['class'])


# In[13]:


#check values for age variable
earnings_data['age'].unique()


# In[14]:


#check values for education years variable
earnings_data['education-years'].unique()


# In[15]:


#check values for marital status variable
earnings_data['marital-status'].unique()


# In[16]:


#check values for occupation variable
earnings_data['occupation'].unique()


# In[17]:


#check values for relationship variable
earnings_data['relationship'].unique()


# In[18]:


#check values for race variable
earnings_data['race'].unique()


# In[19]:


#check values for gender variable
earnings_data['gender'].unique()


# In[20]:


#check values for hours per week variable
earnings_data['hours-per-week'].unique()


# In[21]:


#check values for native country variable
earnings_data['native-country'].unique()


# # Task 1
# ### Plot histograms and box plots of all the numeric features.  For each of these features derive the following…
# ### •	Mean
# ### •	Median
# ### •	Standard deviation
# ### •	Interquartile range
# ### Would you consider any of these feature values to be outliers?  If so which ones?

# In[22]:


#create histogram for Age variable
num_cols = earnings_data.select_dtypes(include=['int64', 'float64']).columns
subset = earnings_data[num_cols]
subset['age'].hist(bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Histogram')
plt.show()


# In[23]:


#create histogram for Education Years variable
subset['education-years'].hist(bins=20)
plt.xlabel('Number of Years of Education')
plt.ylabel('Count')
plt.title('Education Years Histogram')
plt.show()


# In[24]:


#create histogram for Hours per Week variable
subset['hours-per-week'].hist(bins=20)
plt.xlabel('Number of Hours Worked per Week')
plt.ylabel('Count')
plt.title('Hours per Week Histogram')
plt.show()


# In[25]:


#create boxplot for Age variable
plt.boxplot(subset['age'])
plt.ylabel('Age')
plt.title('Age Boxplot')
plt.show()


# In[26]:


#create boxplot for Education Years variable
plt.boxplot(subset['education-years'])
plt.ylabel('Number of Years of Education')
plt.title('Education Years Boxplot')
plt.show()


# In[27]:


#create boxplot for Hours per Week variable
plt.boxplot(subset['hours-per-week'])
plt.ylabel('Number of Hours Worked per Week')
plt.title('Hours per Week Boxplot')
plt.show()


# In[28]:


# print mean, median, standard deivation and IQR for Age variable
print('Mean:', subset.age.mean())
print('Median:', subset.age.median())
print('Standard Deviation:', subset.age.std())
print('IQR:', stats.iqr(subset.age, interpolation='midpoint'))


# In[29]:


# print mean, median, standard deivation and IQR for Education Years variable
print('Mean:', subset['education-years'].mean())
print('Median:', subset['education-years'].median())
print('Standard Deviation:', subset['education-years'].std())
print('IQR:', stats.iqr(subset['education-years'], interpolation='midpoint'))


# In[30]:


# print mean, median, standard deivation and IQR for Hours per Week variable
print('Mean:', subset['hours-per-week'].mean())
print('Median:', subset['hours-per-week'].median())
print('Standard Deviation:', subset['hours-per-week'].std())
print('IQR:', stats.iqr(subset['hours-per-week'], interpolation='midpoint'))


# In[31]:


#Outliers from Age variable using outlier if value > Q3+1.5*IQR or value < Q1-1.5*IQR by Tukey outliers principle
desc = earnings_data.describe()
earnings_data[(earnings_data['age']<desc.loc['25%','age']-1.5*desc.loc['50%', 'age']) | (earnings_data['age']>desc.loc['75%','age']+1.5*desc.loc['50%', 'age'])]


# In[32]:


#Outliers from Education Years variable using outlier if value > Q3+1.5*IQR or value < Q1-1.5*IQR by Tukey outliers principle
desc = earnings_data.describe()
earnings_data[(earnings_data['education-years']<desc.loc['25%','education-years']-1.5*desc.loc['50%', 'education-years']) | (earnings_data['education-years']>desc.loc['75%','education-years']+1.5*desc.loc['50%', 'education-years'])]


# In[33]:


#Outliers from hours per week variable using outlier if value > Q3+1.5*IQR or value < Q1-1.5*IQR by Tukey outliers principle
desc = earnings_data.describe()
earnings_data[(earnings_data['hours-per-week']<(desc.loc['25%','hours-per-week']-1.5*desc.loc['50%', 'hours-per-week'])) | (earnings_data['hours-per-week']>(desc.loc['75%','hours-per-week']+1.5*desc.loc['50%', 'hours-per-week']))]


# # Task 2:
# ### Plot a bubble chart …
# ### •	x-axis: age
# ### •	y-axis: education-year
# ### •	bubble size: hours-per-week
# ### •	bubble colour: pink for female, blue for male

# In[34]:


#create bubble chart

colours = ['blue', 'pink']
ax = sns.relplot(data=earnings_data, x='age', y='education-years', hue='gender', palette=colours, size='hours-per-week', sizes=(40,400), alpha=0.5)
ax.set(title='Bubble chart for age and education years of employees colour coded by gender type and sized by hours worked per week')


# # Task 3
# ### Create a binary classifier that will predict whether someone earns more than 50K or not.
# ### Derive the following:
# ### •	Accuracy
# ### •	Precision
# ### •	Recall
# ### Which feature has the greatest impact on whether they earn 50K or more?  
# ### Which feature is the least important?

# In[35]:


#check distribution of target variable
target = earnings_data.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
    per = v / len(target) * 100
    print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


# In[36]:


# encode categorical variables as binary columns so the model can process them
df_dmy = pd.get_dummies(earnings_data, columns=['marital-status','occupation', 'relationship', 'race', 'gender', 'native-country'], prefix=['ms','occ','rel','race', 'gen', 'nat'])


# In[37]:


# encode the class variable as numeric field
d = {'<=50K':0,'>50K':1}
df_enc = df_dmy.replace({'class':d})
df_enc.head()


# In[38]:


#scale numeric variables using MinMaxScaler
num_subset = df_enc[['age', 'education-years', 'hours-per-week']]
scaler = MinMaxScaler()
num_sub_scaled = pd.DataFrame(scaler.fit_transform(num_subset))


# In[39]:


num_sub_scaled.columns = ['age_scaled', 'education-years_scaled', 'hours-per-week_scaled']


# In[40]:


df = df_enc.join(num_sub_scaled)


# In[41]:


df.head()


# In[42]:


df = df.drop(['age','education-years', 'hours-per-week'], axis=1)


# In[43]:


# split data into predictors, X, and predictant, y
X = df.drop('class', axis=1)
y = df['class']


# In[44]:


# ignore categorical variables as running out of time
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[45]:


# instantiate the random forest classifier model
rf = RandomForestClassifier()


# In[46]:


# fit the model on the training data
rf.fit(X_train, y_train)


# In[47]:


# use the model to make a prediction using the X_test data
y_pred = rf.predict(X_test)


# In[48]:


# evaluate the model using classification report
print('Accuracy score:', metrics.accuracy_score(y_test, y_pred))
print('Precision score:', metrics.precision_score(y_test, y_pred))
print('Recall score:', metrics.recall_score(y_test, y_pred))


# In[49]:


# cross validation for accuracy score of model
rf_cv_scores = cross_val_score(RandomForestClassifier(), X_test, y_test, scoring='accuracy', cv=10)
print(rf_cv_scores)
print(rf_cv_scores.mean())


# In[50]:


# cross validation for precision score of model
rf_cv_scores = cross_val_score(RandomForestClassifier(), X_test, y_test, scoring='precision', cv=10)
print(rf_cv_scores)
print(rf_cv_scores.mean())


# In[51]:


# cross validation for recall score of model
rf_cv_scores = cross_val_score(RandomForestClassifier(), X_test, y_test, scoring='recall', cv=10)
print(rf_cv_scores)
print(rf_cv_scores.mean())


# In[52]:


# check correlation between predictant and predictors to deduce most influential feature 
corr = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, cmap='RdYlGn')


# In[53]:


# take a subset of the most influential features based on heatmap above to drill down 

rel_subset = df[['class', 'gen_Male', 'gen_Female', 'rel_Husband', 'age_scaled', 'education-years_scaled', 'hours-per-week_scaled', 'ms_Married-civ-spouse']]
corr = rel_subset.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True, cmap='RdYlGn')


# # Conclusion
# #### Most influential feature according to the heatmap is ms_Married-civ-spouse
# #### Least influential feature according to the heatmap is native country

# In[ ]:




