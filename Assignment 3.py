#!/usr/bin/env python
# coding: utf-8

# In[242]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[243]:


df=pd.read_csv('/home/ravish/Downloads/sonar.all-data',header=None)


# In[244]:


df.head()


# In[245]:


df.describe()


# In[246]:


df.shape


# In[247]:


df.isnull().sum()


# In[248]:


#X=df.drop(60,axis=1)


# In[249]:


x = df.iloc[:, :-1].values  
y = df.iloc[:, 60].values


# In[250]:


sb.heatmap(X.corr())


# In[251]:


y


# In[252]:


import sklearn as sk


# In[253]:


from sklearn.model_selection import GridSearchCV


# In[254]:


df.dtypes


# In[255]:


#Train Test Split 
from sklearn.model_selection import train_test_split  
X_train, X_test, Y1_train, Y1_test = train_test_split(X, y, test_size = 0.20,random_state=30)


# In[256]:


#Support Vector Classifier Model
from sklearn.svm import SVC  
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train,Y1_train)


# In[257]:


y_pred = svclassifier.predict(X_test)  


# In[258]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y1_test,y_pred))  
print(classification_report(Y1_test,y_pred))  


# In[259]:


print(accuracy_score(Y1_test,y_pred))


# In[260]:


#Parameter tuning using grid search for SVM
param_grid = { 
    'gamma' : [0.1, 1, 10, 100],
    'kernel' :['linear', 'rbf', 'poly'],
}


# In[261]:


CV_svc = GridSearchCV(svclassifier, param_grid=param_grid, cv= 5)
CV_svc.fit(X_train, Y1_train)


# In[262]:


CV_svc.best_params_


# In[263]:


svclassifier1 = SVC(kernel='rbf',gamma=1)  
svclassifier1.fit(X_train,Y1_train)
y_pred_svc=svclassifier1.predict(X_test)


# In[264]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y1_test,y_pred_svc))  
print(classification_report(Y1_test,y_pred_svc))
print(accuracy_score(Y1_test,y_pred_svc))


# In[266]:


#decision tree classifier Model
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train,Y1_train)


# In[267]:


y_pred_dt= classifier.predict(X_test)  


# In[268]:


y_pred_dt


# In[269]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y1_test,y_pred_dt))  
print(classification_report(Y1_test,y_pred_dt))
print(accuracy_score(Y1_test,y_pred_dt))


# In[270]:


#Prameter tuning using grid search for Decision Tree Classifier
param_grid = { 
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[271]:


CV_dt = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 5)
CV_dt.fit(X_train, Y1_train)


# In[272]:


CV_dt.best_params_


# In[273]:


dt1=DecisionTreeClassifier(criterion='entropy',max_depth=7)  


# In[274]:


dt1.fit(X_train,Y1_train)


# In[275]:


y_pred_dt1= dt1.predict(X_test)  


# In[276]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y1_test,y_pred_dt1))  
print(classification_report(Y1_test,y_pred_dt1))
print(accuracy_score(Y1_test,y_pred_dt1))


# In[277]:


#Random forest Classifier
from sklearn.ensemble import RandomForestClassifier 
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,Y1_train)
rfc_pred=rfc.predict(X_test)


# In[278]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y1_test,rfc_pred))  
print(classification_report(Y1_test,rfc_pred))
print(accuracy_score(Y1_test,rfc_pred))


# In[279]:


#Prameter tuning using grid search for Random Forest Classifier
param_grid = { 
    'n_estimators': [50, 100, 200],
    'max_depth' : [4,5,6,7,8,9,10,11],
    'criterion' :['gini', 'entropy']
}


# In[280]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y1_train)


# In[281]:


CV_rfc.best_params_


# In[282]:


rfc1=RandomForestClassifier(n_estimators= 50, max_depth=7, criterion='entropy')


# In[283]:


rfc1.fit(X_train,Y1_train)
rfc1_pred=rfc1.predict(X_test)


# In[284]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y1_test,rfc1_pred))  
print(classification_report(Y1_test,rfc1_pred))
print(accuracy_score(Y1_test,rfc1_pred))


# In[ ]:




