#!/usr/bin/env python
# coding: utf-8

# # Classifier Tests with extracted data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import extractor as ext


# In[2]:


fbase = pd.read_csv('./1_filtered.csv', header=None)
fbase.head()


# In[3]:


extractor = ext.RecordExtractor()
cycles = extractor.read_sample(fbase)
len(cycles)


# In[4]:


# First Sensor, First cycle
cycles[0][0].grip.head()


# In[5]:


# First Sensor, First cycle
cycles[0][0].flexion.head()


# # Generate Dataset based on feature extractors
# 
# 1. SampEn - Sample Entropy
# 2. RMS - Root Mean Square
# 3. WL - Waveform Length
# 4. WAMP - Willison Amplitude
# 5. ApEn - Approximate entropy
# 6. MAV - Mean Absolute Value
# 
# Each cycle will be interpreted as a new data acquisition
# 3 Datasets, one with 299ms, other with 500ms, other with 1 second.
# 
# Usar janela de 2(melhor tempo escolhido) segundos para saber qual o melhor extrator,
# verificar o método de Onset para ver se a marcação é confiável ou não. FAzer estudo a parte.
# 
# Fazer primeiro o feijão com arroz, Usar o tempo base para entender. Checar o onset primeiro
# 
# Usar KNN e SVM pois os dados são poucos.
# 
# Dataset 299ms:
# | SENSOR_0_RMS   |  SENSOR_1_RMS | SENSOR_2_RMS | SENSOR_3_RMS| SENSOR_0_SAMPEN   |  SENSOR_1_SAMPEN | SENSOR_2_SAMPEN | SENSOR_3_SAMPEN |...| CLASS | 
# | --- | ---| ---| ---| ---| ---| ---| ---| ---| ---|
# | 12 | 13 | 45 | 4 | 5 | 23 | 123 | 123|...|GRIP|
# 
# 
# TODO: Do we use the onset algorithm?

# IDEIA:
# 
# Usar dois dados rápidos para análise rápida e depois no meio do movimento da mão verificar a análise com mais tempo e corrigir o movimento

# In[6]:


extractor = ext.RecordExtractor()
cycles = extractor.read_sample(fbase)
np.array(cycles).shape


# In[7]:


import feature_extractors as fe


# In[8]:


def populatedict(data, name, dic:dict):
    data = np.array(data)
    if not dic.get(name+"_RMS"): dic[name+"_RMS"] = []
    if not dic.get(name+"_WAVELEN"): dic[name+"_WAVELEN"] = []
    if not dic.get(name+"_WAMP"): dic[name+"_WAMP"] = []
    if not dic.get(name+"_APPEN"): dic[name+"_APPEN"] = []
    if not dic.get(name+"_SAMPEN"): dic[name+"_SAMPEN"] = []
    if not dic.get(name+"_MAV"): dic[name+"_MAV"] = []

    dic[name+"_RMS"].append(fe.rms(data))
    dic[name+"_WAVELEN"].append(fe.waveformlen(data))
    dic[name+"_WAMP"].append(fe.wamp(data))
    dic[name+"_APPEN"].append(fe.app_entropy(data))
    dic[name+"_SAMPEN"].append(fe.sampen(data))
    dic[name+"_MAV"].append(fe.mav(data))


# In[9]:


newds = {}
newds_offsetted = {}
newds["class"] = []
newds_offsetted["class"] = []

for j in range(4):
    for i in range(5):
        lower_limit = int(extractor.to_ms(100))
        upper_limit = int(extractor.to_ms(299)) # TODO: Create two databases, one with the offset and the other without

        populatedict(cycles[j][i].rest[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].rest[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].extension[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].extension[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].flexion[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].flexion[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].ulnar_deviation[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].ulnar_deviation[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].radial_deviation[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].radial_deviation[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].grip[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].grip[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].finger_abduction[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].finger_abduction[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].finger_adduction[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].finger_adduction[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].supination[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].supination[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

        populatedict(cycles[j][i].pronation[:upper_limit], "SENSOR" + str(j), newds)
        populatedict(cycles[j][i].pronation[lower_limit:lower_limit+upper_limit], "SENSOR" + str(j), newds_offsetted)

for i in range(5):
    newds["class"].append("rest")
    newds_offsetted["class"].append("rest")
    
for i in range(5):
    newds["class"].append("extension")
    newds_offsetted["class"].append("extension")

for i in range(5):
    newds["class"].append("flexion")
    newds_offsetted["class"].append("flexion")

for i in range(5):
    newds["class"].append("ulnar_deviation")
    newds_offsetted["class"].append("ulnar_deviation")

for i in range(5):
    newds["class"].append("radial_deviation")
    newds_offsetted["class"].append("radial_deviation")

for i in range(5):
    newds["class"].append("grip")
    newds_offsetted["class"].append("grip")

for i in range(5):
    newds["class"].append("finger_abduction")
    newds_offsetted["class"].append("finger_abduction")

for i in range(5):
    newds["class"].append("finger_adduction")
    newds_offsetted["class"].append("finger_adduction")

for i in range(5):
    newds["class"].append("supination")
    newds_offsetted["class"].append("supination")

for i in range(5):
    newds["class"].append("pronation")
    newds_offsetted["class"].append("pronation")

print(len(newds["class"]))
print(len(newds["SENSOR0_MAV"]))


# In[17]:


pd.DataFrame(newds).head()


# In[11]:


pd.DataFrame(newds).head(10)


# In[12]:


pd.DataFrame(newds_offsetted).head(10)


# ### Code encapsulated on `dataset_creation.py`

# In[13]:


import dataset_creation as dtc

ds, offsetted_ds = dtc.create_dataset(fbase)


# In[16]:


ds.head(10)


# ## Now import more data

# In[19]:


fbase2 = pd.read_csv('./2_filtered.csv', header=None)
fbase3 = pd.read_csv('./3_filtered.csv', header=None)


ds_2, _ = dtc.create_dataset(fbase2)
ds_3, _ = dtc.create_dataset(fbase3)


# In[21]:


full_ds = pd.concat([ds, ds_2, ds_3])
full_ds


# In[23]:


full_ds.columns


# In[64]:


import numpy as np

import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.neighbors import KNeighborsClassifier


# ## RMS Test

# In[24]:


class_column = "class"
train_columns = [f"SENSOR{i}_RMS" for i in range(4)]

print(train_columns)


# In[63]:


X = full_ds[train_columns]
Y = full_ds[class_column]


# In[57]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2)

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)

labeled_Y_test = Y_test
Y_test = encoder.transform(Y_test)


# In[58]:


model = KNeighborsClassifier()
model.fit(X_train, Y_train)


# In[62]:


preds_valid = model.predict(X_test)
encoder.inverse_transform(preds_valid)


# In[60]:


score_valid = mean_absolute_error(Y_test, preds_valid)
score_valid


# In[65]:


accuracy_score(Y_test, preds_valid)


# ### Defining function to facilitate

# In[66]:


def get_score_and_accuracy(X, Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2)

    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)

    labeled_Y_test = Y_test
    Y_test = encoder.transform(Y_test)
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    preds_valid = model.predict(X_test)
    return mean_absolute_error(Y_test, preds_valid), accuracy_score(Y_test, preds_valid)


# ## Appen Test

# In[67]:


class_column = "class"
train_columns = [f"SENSOR{i}_APPEN" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]


# In[70]:


get_score_and_accuracy(X, Y)


# ## Sampen Test

# In[71]:


class_column = "class"
train_columns = [f"SENSOR{i}_SAMPEN" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]

get_score_and_accuracy(X, Y)


# ## MAV Test

# In[73]:


class_column = "class"
train_columns = [f"SENSOR{i}_MAV" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]

get_score_and_accuracy(X, Y)


# ## WAMP Test

# In[74]:


class_column = "class"
train_columns = [f"SENSOR{i}_WAMP" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]

get_score_and_accuracy(X, Y)


# ## WAVELEN Test

# In[75]:


class_column = "class"
train_columns = [f"SENSOR{i}_WAVELEN" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]

get_score_and_accuracy(X, Y)


# # Results
# Bad.
# 
# Maybe we need to use the on-set algorithm, we are having really bad results

# Another tests with svm:

# In[76]:


from sklearn.svm import SVC


def get_score_and_accuracy_svm(X, Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2)
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)

    labeled_Y_test = Y_test
    Y_test = encoder.transform(Y_test)
    model = SVC()
    model.fit(X_train, Y_train)
    preds_valid = model.predict(X_test)
    return mean_absolute_error(Y_test, preds_valid), accuracy_score(Y_test, preds_valid)


# In[77]:


class_column = "class"
train_columns = [f"SENSOR{i}_WAMP" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]

get_score_and_accuracy_svm(X, Y)


# In[78]:


class_column = "class"
train_columns = [f"SENSOR{i}_MAV" for i in range(4)]

X = full_ds[train_columns]
Y = full_ds[class_column]

get_score_and_accuracy(X, Y)


# In[ ]:




