
# coding: utf-8

# In[ ]:



# coding: utf-8

# # CoDeepNEAT demo
# ## CS081 project checkpoint demo
# ### Harsha Uppli, Alan Zhao, Gabriel Meyer-Lee
#
# The following notebook demonstrates using CoDeepNEAT to solve the Penn Tree Bank

# In[7]:


from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import preprocessing
from keras import backend as K
from math import pi, floor
from random import random
import pickle
import numpy as np
import keras
import collections
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM



# In[8]:


max_words = 10000
print('Loading data...')
(x_train_all, y_train_all), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train_all), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train_all) + 1
print(num_classes, 'classes')



x_train_all = preprocessing.sequence.pad_sequences(x_train_all, maxlen=30)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=30)
y_train_all = keras.utils.to_categorical(y_train_all, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# In[ ]:


model = Sequential()

model.add(LSTM(int(max_words*1.5), input_shape=(max_words, num_classes)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# In[ ]:


# Train model
model.fit(x_train_all, y_train_all, batch_size=128, nb_epoch=5)

# Evaluate model
score, acc = model.evaluate(x_test, y_test, batch_size=128)

print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)
