#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


df=pd.read_csv('MovieReviewTrainingDatabase.csv',encoding="utf8",)
df


# In[24]:


def loweer(row):
    low = row['review'].lower()
    return low
df['review'] = df.apply(loweer, axis=1)
df


# In[25]:


import re
def punc(row):
    low = re.sub("[^A-Za-z]"," ",row['review'])
    return low
df['review'] = df.apply(punc, axis=1)
df


# In[26]:


df


# In[27]:


df=df.sample(frac=0.4, replace=True, random_state=1)


# In[28]:


review = df['review']
review


# In[29]:


vectorizer = CountVectorizer(stop_words='english')
review_vecctor = vectorizer.fit_transform(review)
review_vecctor.toarray()


# In[30]:


vectorizer.get_feature_names()


# In[31]:


ddf = pd.DataFrame(review_vecctor.toarray(), columns=vectorizer.get_feature_names())
ddf


# In[32]:


x = ddf.drop('sentiment', axis=1)
y = df['sentiment']


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)


# In[35]:


from sklearn.preprocessing import StandardScaler


# In[36]:


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, confusion_matrix


# In[38]:


models = {
    'LR': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    #'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'NB':GaussianNB()

}


# In[39]:


for name, model in models.items():
    print(f'Model: {name}')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    #print(f'Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print('-'*80)


# In[19]:


#n_estimators = [5,20,50,100] # number of trees in the random forest
#max_features = ['auto', 'sqrt'] # number of features in consideration at every split
#max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
#min_samples_split = [2, 6, 10] # minimum sample number to split a node
#min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
#bootstrap = [True, False] # method used to sample data points

#random_grid = {'n_estimators': n_estimators,

#'max_features': max_features,

# 'max_depth': max_depth,
#'min_samples_split': min_samples_split,

#'min_samples_leaf': min_samples_leaf,

#'bootstrap': bootstrap}


# In[20]:


#Importing Random Forest Classifier from the sklearn.ensemble
#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor()


# In[21]:


##Importing Random Forest Classifier from the sklearn.ensemble
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier()
#from sklearn.model_selection import RandomizedSearchCV
##rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
##n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[22]:


#rf_random.fit(x_train, y_train)


# In[23]:


#print ('Random grid: ', random_grid, '\n')
#print the best parameters
#print ('Best Parameters: ', rf_random.best_params_, ' \n')


# In[40]:


randmf = RandomForestClassifier(n_estimators = 200, min_samples_split = 5, min_samples_leaf= 1, max_features = 'sqrt', max_depth= 100, bootstrap=False) 
randmf.fit( x_train, y_train) 
y_pred = randmf.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
#print(f'Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
#print(f'Testing MAE: {mean_absolute_error(y_test, y_pred)}')
#print('-'*50)


# In[25]:


model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

