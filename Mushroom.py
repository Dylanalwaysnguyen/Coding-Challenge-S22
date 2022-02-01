#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd


# In[ ]:


import numpy as np


# In[ ]:


import seaborn as sns


# In[5]:


df = pd.read_csv("https://raw.githubusercontent.com/ACM-Research/Coding-Challenge-S22/main/mushrooms.csv")


# In[18]:


df.head()


# In[8]:


inputs = df.drop('class',axis='columns')
target = df['class']


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[40]:


df.info()


# In[22]:


le_capshape = LabelEncoder()
le_capsurface = LabelEncoder()
le_capcolor = LabelEncoder()
le_bruises = LabelEncoder()
le_odor = LabelEncoder()
le_gillattachment = LabelEncoder()
le_gillspacing = LabelEncoder()
le_gillsize = LabelEncoder()
le_gillcolor = LabelEncoder()
le_stalkshape = LabelEncoder()
le_stalkroot = LabelEncoder()
le_stalksurfacebelowring = LabelEncoder()
le_stalkcolorabovering = LabelEncoder()
le_stalkcolorbelowring = LabelEncoder()
le_veiltype = LabelEncoder()
le_veilcolor = LabelEncoder()
le_ringnumber = LabelEncoder()
le_ringtype = LabelEncoder()
le_sporeprintcolor = LabelEncoder()
le_population = LabelEncoder()
le_habitat = LabelEncoder()


# In[41]:


df.columns


# In[23]:


inputs['capshape_n'] = le_capshape.fit_transform(inputs['cap-shape'])
inputs['capsurface_n'] = le_capsurface.fit_transform(inputs['cap-surface'])
inputs['capcolor_n'] = le_capcolor.fit_transform(inputs['cap-color'])
inputs['bruises_n'] = le_bruises.fit_transform(inputs['bruises'])
inputs['odor_n'] = le_odor.fit_transform(inputs['odor'])
inputs['gillattachment_n'] = le_gillattachment.fit_transform(inputs['gill-attachment'])
inputs['gillspacing_n'] = le_gillspacing.fit_transform(inputs['gill-spacing'])
inputs['gillsize_n'] = le_gillsize.fit_transform(inputs['gill-size'])
inputs['gillcolor_n'] = le_gillcolor.fit_transform(inputs['gill-color'])
inputs['stalkshape_n'] = le_stalkshape.fit_transform(inputs['stalk-shape'])
inputs['stalkroot_n'] = le_stalkroot.fit_transform(inputs['stalk-root'])
inputs['stalksurfacebelowring_n'] = le_stalksurfacebelowring.fit_transform(inputs['stalk-surface-below-ring'])
inputs['stalkcolorabovering_n'] = le_stalkcolorabovering.fit_transform(inputs['stalk-color-above-ring'])
inputs['stalkcolorbelowring_n'] = le_stalkcolorbelowring.fit_transform(inputs['stalk-color-below-ring'])
inputs['veiltype_n'] = le_veiltype.fit_transform(inputs['veil-type'])
inputs['veilcolor_n'] = le_veilcolor.fit_transform(inputs['veil-color'])
inputs['ringnumber_n'] = le_ringnumber.fit_transform(inputs['ring-number'])
inputs['ringtype_n'] = le_ringtype.fit_transform(inputs['ring-type'])
inputs['sporeprintcolor_n'] = le_sporeprintcolor.fit_transform(inputs['spore-print-color'])
inputs['population_n'] = le_population.fit_transform(inputs['population'])
inputs['habitat_n'] = le_habitat.fit_transform(inputs['habitat'])
inputs.head()


# In[31]:


inputs_n = inputs.drop(['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'],axis='columns')
inputs_n


# In[25]:


from sklearn import tree


# In[32]:


model.score(inputs_n,target)


# In[44]:


from sklearn.model_selection import train_test_split


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(inputs_n,target,test_size=0.33)


# In[48]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


model = tree.DecisionTreeClassifier()


# In[60]:


model.fit(X_train,y_train)


# In[ ]:


model.fit(X_train,y_train)


# In[54]:


from sklearn.tree import DecisionTreeRegressor


# In[49]:


model.get_params()


# In[50]:


X_test


# In[62]:


predictions = model.predict(X_test)
predictions


# In[73]:


model.score(X_test,y_test)


# In[74]:


model.predict([X_train[2]]) == y_train[2]


# In[ ]:




