#!/usr/bin/env python
# coding: utf-8

# # Autoencoder Approach to Feature Classification
# The following is an attempt to improve the previous author's neural network classification approach by using autoencoders.  
# The code from the original neural network approach can be found in the file "prepare_raw_data.py".  
# It was copied into this jupyter notebook and updated to use Python 3.7 and Tensorflow 2 (which includes Keras).

# Table of contents:  
# [Preparing the Data and Autoencoders](#Preparing-the-Data-and-Autoencoders)  
# [Undercomplete Autoencoders](#Undercomplete-Autoencoders)  
# [Sparse Autoencoders](#Sparse-Autoencoders)  
# [Deep Autoencoders](#Deep-Autoencoders)  

# In[19]:


import datareader # made by the previous author for reading the collected data
import dataextractor # same as above
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import json


# This is the original author's code, just copied into separate cells of this jupyter notebook:

# In[20]:


def get_busy_vs_relax_timeframes(path, ident, seconds):
    """Returns raw data from either 'on task' or 'relax' time frames and their class (0 or 1).
    TODO: join functions"""

    dataread = datareader.DataReader(path, ident)  # initialize path to data
    data = dataread.read_grc_data()  # read from files
    samp_rate = int(round(len(data[1]) / max(data[0])))
    cog_res = dataread.read_cognitive_load_study(str(ident) + '-primary-extract.txt')

    tasks_data = np.empty((0, seconds*samp_rate))
    tasks_y = np.empty((0, 1))

    busy_n = dataread.get_data_task_timestamps(return_indexes=True)
    relax_n = dataread.get_relax_timestamps(return_indexes=True)

    for i in cog_res['task_number']:
        task_num_table = i - 225  # 0 - 17

        ### task versus relax (1 sample each)
        dataextract = dataextractor.DataExtractor(data[0][busy_n[task_num_table][0]:busy_n[task_num_table][1]],
                                                  data[1][busy_n[task_num_table][0]:busy_n[task_num_table][1]],
                                                  samp_rate)

        dataextract_relax = dataextractor.DataExtractor(data[0][relax_n[task_num_table][0]:relax_n[task_num_table][1]],
                                                        data[1][relax_n[task_num_table][0]:relax_n[task_num_table][1]],
                                                        samp_rate)
        try:
            tasks_data = np.vstack((tasks_data, dataextract.y[-samp_rate * seconds:]))
            tasks_y = np.vstack((tasks_y, 1))
            tasks_data = np.vstack((tasks_data, dataextract_relax.y[-samp_rate * seconds:]))
            tasks_y = np.vstack((tasks_y, 0))
        except ValueError:
            print(ident)  # ignore short windows

    return tasks_data, tasks_y


# In[21]:


def get_engagement_increase_vs_decrease_timeframes(path, ident, seconds):
    """Returns raw data from either engagement 'increase' or 'decrease' time frames and their class (0 or 1).
    TODO: join functions"""

    dataread = datareader.DataReader(path, ident)  # initialize path to data
    data = dataread.read_grc_data()  # read from files
    samp_rate = int(round(len(data[1]) / max(data[0])))
    cog_res = dataread.read_cognitive_load_study(str(ident) + '-primary-extract.txt')

    tasks_data = np.empty((0, seconds * samp_rate))
    tasks_y = np.empty((0, 1))

    busy_n = dataread.get_data_task_timestamps(return_indexes=True)
    relax_n = dataread.get_relax_timestamps(return_indexes=True)

    for i in cog_res['task_number']:
        task_num_table = i - 225  # 0 - 17

        ### engagement increase / decrease
        if task_num_table == 0:
            continue
        mid = int((relax_n[task_num_table][0] + relax_n[task_num_table][1])/2)
        length = int(samp_rate*30)
        for j in range(10):
            new_end = int(mid-j*samp_rate)

            new_start2 = int(mid+j*samp_rate)

            dataextract_decrease = dataextractor.DataExtractor(data[0][new_end - length:new_end],
                                                               data[1][new_end-length:new_end],
                                                               samp_rate)

            dataextract_increase = dataextractor.DataExtractor(data[0][new_start2: new_start2 + length],
                                                               data[1][new_start2: new_start2 + length], samp_rate)

            try:
                tasks_data = np.vstack((tasks_data, dataextract_increase.y))
                tasks_y = np.vstack((tasks_y, 1))
                tasks_data = np.vstack((tasks_data, dataextract_decrease.y))
                tasks_y = np.vstack((tasks_y, 0))
            except ValueError:
                print(ident)  # ignore short windows

    return tasks_data, tasks_y


# In[22]:


def get_task_complexities_timeframes(path, ident, seconds):
    """Returns raw data along with task complexity class.
    TODO: join functions. Add parameter to choose different task types and complexities"""

    dataread = datareader.DataReader(path, ident)  # initialize path to data
    data = dataread.read_grc_data()  # read from files
    samp_rate = int(round(len(data[1]) / max(data[0])))
    cog_res = dataread.read_cognitive_load_study(str(ident) + '-primary-extract.txt')

    tasks_data = np.empty((0, seconds*samp_rate))
    tasks_y = np.empty((0, 1))

    busy_n = dataread.get_data_task_timestamps(return_indexes=True)
    relax_n = dataread.get_relax_timestamps(return_indexes=True)

    for i in cog_res['task_number']:
        task_num_table = i - 225  # 0 - 17

        ### task complexity classification
        if cog_res['task_complexity'][task_num_table] == 'medium':
            continue
        # if cog_res['task_label'][task_num_table] == 'FA' or cog_res['task_label'][task_num_table] == 'HP':
        #     continue
        if cog_res['task_label'][task_num_table] != 'NC':
            continue
        map_compl = {
            'low': 0,
            'medium': 2,
            'high': 1
        }
        for j in range(10):
            new_end = int(busy_n[task_num_table][1] - j * samp_rate)
            new_start = int(new_end - samp_rate*30)
            dataextract = dataextractor.DataExtractor(data[0][new_start:new_end],
                                                      data[1][new_start:new_end], samp_rate)
            try:
                tasks_data = np.vstack((tasks_data, dataextract.y))
                tasks_y = np.vstack((tasks_y, map_compl.get(cog_res['task_complexity'][task_num_table])))
            except ValueError:
                print(ident)

    return tasks_data, tasks_y


# In[23]:


def get_TLX_timeframes(path, ident, seconds):
    """Returns raw data along with task load index class.
    TODO: join functions. Add parameter to choose different task types and complexities"""

    dataread = datareader.DataReader(path, ident)  # initialize path to data
    data = dataread.read_grc_data()  # read from files
    samp_rate = int(round(len(data[1]) / max(data[0])))
    cog_res = dataread.read_cognitive_load_study(str(ident) + '-primary-extract.txt')

    tasks_data = np.empty((0, seconds*samp_rate))
    tasks_y = np.empty((0, 1))

    busy_n = dataread.get_data_task_timestamps(return_indexes=True)
    relax_n = dataread.get_relax_timestamps(return_indexes=True)

    for i in cog_res['task_number']:
        task_num_table = i - 225  # 0 - 17

        ### task load index
        if cog_res['task_complexity'][task_num_table] == 'medium' or cog_res['task_label'][task_num_table] != 'PT':
            continue
        for j in range(10):
            new_end = int(busy_n[task_num_table][1] - j * samp_rate)
            new_start = int(new_end - samp_rate*30)
            dataextract = dataextractor.DataExtractor(data[0][new_start:new_end],
                                                      data[1][new_start:new_end], samp_rate)
            try:
                tasks_data = np.vstack((tasks_data, dataextract.y))
                tasks_y = np.vstack((tasks_y, cog_res['task_load_index'][task_num_table]))
            except ValueError:
                print(ident)

    return tasks_data, tasks_y


# In[24]:


def get_data_from_idents(path, idents, seconds):
    """Go through all user data and take out windows of only <seconds> long time frames,
    along with the given class (from 'divide_each_task' function).
    """
    samp_rate = 43  # hard-coded sample rate
    data, ys = np.empty((0, samp_rate*seconds)), np.empty((0, 1))
    for i in idents:
        x, y = get_busy_vs_relax_timeframes(path, i, seconds) # either 'get_busy_vs_relax_timeframes',
        # get_engagement_increase_vs_decrease_timeframes, get_task_complexities_timeframes or get_TLX_timeframes
        # TODO: ^ modify, so that different functions can be accessible by parameter
        data = np.vstack((data, x))
        ys = np.vstack((ys, y))
    return data, ys


# In[25]:


def model_build_reg():
    """Neural network model for regression problem."""

    print('Build model...')

    data_dim = 1

    # Convolution
    kernel_size = 10
    filters = 64
    strides = 4
    # pooling
    pool_size = 4

    # LSTM
    lstm_output_size = 256

    model = Sequential()
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=strides))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=strides))
    model.add(MaxPooling1D(pool_size=pool_size))
    # #
    model.add(Dropout(0.25))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])

    return model


# In[26]:


def model_build_multi():
    """Neural network model for multi class problem."""

    print('Build model...')

    # Convolution
    kernel_size = 10
    filters = 64
    strides = 4
    # pooling
    pool_size = 4

    # LSTM
    lstm_output_size = 256

    model = Sequential()
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=strides))

    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=strides))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(LSTM(lstm_output_size))
    # model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[27]:


def model_build_bin():
    """Neural network model for binary problem."""

    print('Build model...')

    # Convolution
    kernel_size = 200
    filters = 64
    strides = 4
    # pooling
    pool_size = 4
    # LSTM
    lstm_output_size = 256

    model = Sequential()
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=strides))

    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=strides))
    model.add(MaxPooling1D(pool_size=pool_size))
    # #
    model.add(Dropout(0.25))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[28]:


def model_train(model, x_train, y_train, batch_size, epochs, x_valid, y_valid, x_test, y_test):
    """Train model with the given training, validation, and test set, with appropriate batch size and # epochs."""
    print('Train...')
    epoch_data = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return score, acc, epoch_data


# In[29]:


def sequence_padding(x, maxlen):
    """Pad sequences (all have to be same length)."""
    print('Pad sequences (samples x time)')
    return sequence.pad_sequences(x, maxlen=maxlen, dtype=np.float)


# In[30]:


def __main__():
    path = '../../StudyData/' #'E:/signal_acquiring/'

#     idents = ['1mpau']
    idents = ['2gu87', 'iz2ps', '1mpau', '7dwjy', '7swyk', '94mnx', 'bd47a', 'c24ur', 'ctsax', 'dkhty', 'e4gay',
              'ef5rq', 'f1gjp', 'hpbxa', 'pmyfl', 'r89k1', 'tn4vl', 'td5pr', 'gyqu9', 'fzchw', 'l53hg', '3n2f9',
              '62i9y']

    # idents = ['7dwjy', 'bd47a', 'f1gjp', 'hpbxa', 'l53hg', 'tn4vl']
    # idents = ['94mnx', 'fzchw', 'ef5rq', 'iz2ps', 'c24ur', 'td5pr', '3n2f9', 'r89k1']

    user_epoch_data = {}
    seconds = 30  # time window length

    # leave out person out validation
    for ident in range(1):#len(idents)):
        train_idents = [x for i, x in enumerate(idents) if i != ident]
        validation_idents = [idents[ident]]
        test_idents = [idents[ident]]

        x_train, y_train = get_data_from_idents(path, train_idents, seconds)
        x_valid, y_valid = get_data_from_idents(path, validation_idents, seconds)
        x_test, y_test = get_data_from_idents(path, test_idents, seconds)

        x_train = x_train.reshape(-1, x_train[0].shape[0], 1)
        x_valid = x_valid.reshape(-1, x_valid[0].shape[0], 1)
        x_test = x_test.reshape(-1, x_test[0].shape[0], 1)
        
        # Training
        batch_size = 128
        epochs = 100

        model = model_build_bin()
        # model = model_build_multi()
        # model = model_build_reg()
        
        sc, curr_acc, epoch_data = model_train(model, x_train, y_train, batch_size, epochs, x_valid, y_valid, x_test,
                                               y_test)

        epoch_data.history['user_id'] = idents[ident]
        user_epoch_data[idents[ident]] = epoch_data.history

        jsonformat = json.dumps(str(user_epoch_data))
        f = open("./AEOutput/compl2_2_NC_kernel_200_pool_4_filter_64_strides_4_lstm_256.json", "w")
        f.write(jsonformat)
        f.close()


# In[31]:


__main__()


# # Preparing the Data and Autoencoders

# In[64]:


#Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64') # call this, to set keras to use float64 to avoid a warning message


# ### Prepare the Data
# Read the data from files and store it into arrays.

# In[65]:


# Mostly a copy of the code in __main__(), used for reading the data into an array
seconds = 30  # time window length
idents = ['2gu87', 'iz2ps', '1mpau', '7dwjy', '7swyk', '94mnx', 'bd47a', 'c24ur', 'ctsax', 'dkhty', 'e4gay',
              'ef5rq', 'f1gjp', 'hpbxa', 'pmyfl', 'r89k1', 'tn4vl', 'td5pr', 'gyqu9', 'fzchw', 'l53hg', '3n2f9',
              '62i9y']
path = '../../StudyData/'

train_idents = idents[:-1]
validation_idents = [idents[-1]]
test_idents = [idents[-1]]

x_train, y_train = get_data_from_idents(path, train_idents, seconds)
x_valid, y_valid = get_data_from_idents(path, validation_idents, seconds)
x_test, y_test = get_data_from_idents(path, test_idents, seconds)

# x_train = x_train.reshape(-1, x_train[0].shape[0], 1)
# x_valid = x_valid.reshape(-1, x_valid[0].shape[0], 1)
# x_test = x_test.reshape(-1, x_test[0].shape[0], 1)

print("x_train shape:", x_train.shape, "  x_test shape:", x_test.shape)


# ### Preprocess the Data
# Prepare two versions of the data:
# - Normalize original data (**x_\***): reffered to as **normalized data** from now on,
# - subsampled, filtered, then normalized data (**x2_\***): reffered to as **shortened data** form now on.

# In[66]:


from scipy.ndimage.filters import gaussian_filter1d

step = 4 # take each step-th element of the array
sigma = 1 # sigma for gaussian filter

def convolve(x): # helper function for applying along axis
    tmp = gaussian_filter1d(x, sigma=sigma, mode="mirror")
    tmp = normalize(tmp)
    return tmp

def normalize(x): # helper function for applying along axis
    #normalize the data
    tmp = x
    tmp_min = np.min(tmp)
    tmp_max = np.max(tmp)
    top_norm = tmp-tmp_min
    bot_norm = tmp_max-tmp_min
    if (bot_norm == 0): # avoid division by 0
        bot_norm = 1
    tmp = top_norm/bot_norm
    return tmp

# Prepare another set of data that is subsampled, filtered and normalized
# Use np.apply_along_axis to apply the above function to each row of the array separately
x2_train = x_train[:,1::step]
x2_train = np.apply_along_axis(convolve, 1, x2_train)

x2_valid = x_valid[:,1::step]
x2_valid = np.apply_along_axis(convolve, 1, x2_valid)

x2_test = x_test[:,1::step]
x2_test = np.apply_along_axis(convolve, 1, x2_test)

# Normalize original (normalized) data
# Use np.apply_along_axis to apply the above function to each row of the array separately
x_train = np.apply_along_axis(normalize, 1, x_train)
x_valid = np.apply_along_axis(normalize, 1, x_valid)
x_test = np.apply_along_axis(normalize, 1, x_test)

print("Normalized data:", x_test.shape, "  Shortened data:", x2_test.shape)


# In[67]:


#plot n samples to compare the normalized data to the shortened data
n = 33

#plot some normalized data values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_train[i])

#plot some shortened data values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x2_train[i])


# ### Undercomplete Autoencoders
# from https://blog.keras.io/building-autoencoders-in-keras.html

# #### Undercomplete Autoencoder - Normalized Input Data

# Build the autoencoder:

# In[68]:


# Simplest possible autoencoder from https://blog.keras.io/building-autoencoders-in-keras.html

# this is the size of our encoded representations
encoding_dim = 64

# this is our input placeholder
input_data = Input(shape=x_test[0].shape, name="input")
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu', name="encoded")(input_data)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(x_test[0].shape[0], activation='sigmoid', name="decoded")(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_data, decoded, name="undercomplete_ae")


# Print the summary of the autoencoder:

# In[69]:


autoencoder.summary()
keras.utils.plot_model(autoencoder, "./AEOutput/undercomplete_ae.png", show_shapes=True)


# Compile and train the model:

# In[70]:


# Compile the model

# autoencoder.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam')

# Training
batch_size = 256
epochs = 1000

# Fit the model
autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))


# Plot some input data and the corresponding reconstructed data: 

# In[71]:


x_pred = autoencoder.predict(x_train)


#plot n samples to compare the input and reconstruction
n = 33

#plot the input values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_train[i])
    
#plot the reconstructed values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_pred[i])


# Combine the encoder layer and the classification network from above:  
# <i><b>Work in progress!</b> This is just a test, not yet done.</i>

# In[72]:


# leave out person out validation
user_epoch_data = {}
seconds = 30  # time window length

for ident in range(3):
    train_idents = [x for i, x in enumerate(idents) if i != ident]
    validation_idents = [idents[ident]]
    test_idents = [idents[ident]]

    xt_train, yt_train = get_data_from_idents(path, train_idents, seconds)
    xt_valid, yt_valid = get_data_from_idents(path, validation_idents, seconds)
    xt_test, yt_test = get_data_from_idents(path, test_idents, seconds)

    xt_train = np.apply_along_axis(normalize, 1, xt_train)
    xt_valid = np.apply_along_axis(normalize, 1, xt_valid)
    xt_test = np.apply_along_axis(normalize, 1, xt_test)

    # AE Training
    batch_size = 256
    epochs = 1000

    # this is the size of our encoded representations
    encoding_dim = 64
    
    input_data = Input(shape=xt_test[0].shape, name="input")
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu', name="encoded")(input_data)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(xt_test[0].shape[0], activation='sigmoid', name="decoded")(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded, name="simple_ae")
    
    autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    print("\n\nAE TRAINING: ", ident)
    sc, curr_acc, epoch_data = model_train(autoencoder, xt_train, xt_train, batch_size, epochs, xt_valid, xt_valid, xt_test,
                                           xt_test)

    model = autoencoder.get_layer("encoded").output
    # Convolution
    kernel_size = 16
    filters = 16
    strides = 1
    # pooling
    pool_size = 2
    # LSTM
    lstm_output_size = 32

    model = layers.Reshape((-1, 1), input_shape=(encoded.shape)) (model)
    model = Dropout(0.25) (model)

    model = Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=strides) (model)
    model = MaxPooling1D(pool_size=pool_size) (model)

    model = Dropout(0.25) (model)
    model = LSTM(lstm_output_size, activation='sigmoid') (model)
    model = Dense(1, activation='sigmoid') (model)

    model = Model(inputs=autoencoder.inputs, outputs=model)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Classifier Training
    batch_size = 256
    epochs = 100
    
    print("\n\nCLASSIFICATION TRAINING: ", ident)
    sc, curr_acc, epoch_data = model_train(model, xt_train, yt_train, batch_size, epochs, xt_valid, yt_valid, xt_test,
                                           yt_test)

    epoch_data.history['user_id'] = idents[ident]
    user_epoch_data[idents[ident]] = epoch_data.history

    jsonformat = json.dumps(str(user_epoch_data))
    f = open("./AEOutput/compl2_2_NC_kernel_200_pool_4_filter_64_strides_4_lstm_256.json", "w")
    f.write(jsonformat)
    f.close()


# In[ ]:





# #### Undercomplete Autoencoder - Shortened Input Data

# Build the autoencoder:

# In[73]:


# Simplest possible autoencoder from https://blog.keras.io/building-autoencoders-in-keras.html

# this is the size of our encoded representations
encoding_dim = 64

# this is our input placeholder
input_data2 = Input(shape=x2_test[0].shape, name="input2")
# "encoded" is the encoded representation of the input
encoded2 = Dense(encoding_dim, activation='relu', name="encoded2")(input_data2)
# "decoded" is the lossy reconstruction of the input
decoded2 = Dense(x2_test[0].shape[0], activation='sigmoid', name="decoded2")(encoded2)

# this model maps an input to its reconstruction
autoencoder2 = Model(input_data2, decoded2, name="undercomplete_ae2")


# Print the summary of the autoencoder:

# In[74]:


autoencoder2.summary()
keras.utils.plot_model(autoencoder2, "./AEOutput/undercomplete_ae2.png", show_shapes=True)


# Compile and train the model:

# In[75]:


# Compile the model

# autoencoder2.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

autoencoder2.compile(loss='binary_crossentropy',
              optimizer='adam')

# Training
batch_size = 256
epochs = 1000

# Fit the model
autoencoder2.fit(x2_train, x2_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x2_test, x2_test))


# Plot some input data and the corresponding reconstructed data: 

# In[76]:


x2_pred = autoencoder2.predict(x2_train)


#plot n samples to compare
n = 33

#plot the original values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x2_train[i])
    
#plot the reconstructed values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x2_pred[i])


# Combine the encoder layer and the classification network from above:  
# <i><b>Work in progress!</b> This is just a test, not yet done.</i>

# In[79]:


# leave out person out validation
user_epoch_data = {}
seconds = 30  # time window length

for ident in range(3):
    train_idents = [x for i, x in enumerate(idents) if i != ident]
    validation_idents = [idents[ident]]
    test_idents = [idents[ident]]

    xt_train, yt_train = get_data_from_idents(path, train_idents, seconds)
    xt_valid, yt_valid = get_data_from_idents(path, validation_idents, seconds)
    xt_test, yt_test = get_data_from_idents(path, test_idents, seconds)

    x2t_train = xt_train[:,1::step]
    x2t_train = np.apply_along_axis(convolve, 1, x2t_train)

    x2t_valid = xt_valid[:,1::step]
    x2t_valid = np.apply_along_axis(convolve, 1, x2t_valid)

    x2t_test = xt_test[:,1::step]
    x2t_test = np.apply_along_axis(convolve, 1, x2t_test)

    # AE Training
    batch_size = 256
    epochs = 500

    # this is the size of our encoded representations
    encoding_dim = 64

    # this is our input placeholder
    input_data2 = Input(shape=x2t_test[0].shape, name="input2")
    # "encoded" is the encoded representation of the input
    encoded2 = Dense(encoding_dim, activation='relu', name="encoded2")(input_data2)
    # "decoded" is the lossy reconstruction of the input
    decoded2 = Dense(x2t_test[0].shape[0], activation='sigmoid', name="decoded2")(encoded2)

    # this model maps an input to its reconstruction
    autoencoder2 = Model(input_data2, decoded2, name="simple_ae2")
    
    autoencoder2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    print("\n\nAE TRAINING: ", ident)
    sc, curr_acc, epoch_data = model_train(autoencoder2, x2t_train, x2t_train, batch_size, epochs, x2t_valid, x2t_valid, x2t_test,
                                           x2t_test)

    model = autoencoder2.get_layer("encoded2").output
    # Convolution
    kernel_size = 16
    filters = 16
    strides = 1
    # pooling
    pool_size = 2
    # LSTM
    lstm_output_size = 32

    model = layers.Reshape((-1, 1), input_shape=(encoded2.shape)) (model)
    model = Dropout(0.25) (model)

    model = Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=strides) (model)
    model = MaxPooling1D(pool_size=pool_size) (model)

    model = Dropout(0.25) (model)
    model = LSTM(lstm_output_size, activation='sigmoid') (model)
    model = Dense(1, activation='sigmoid') (model)

    model = Model(inputs=autoencoder2.inputs, outputs=model)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Classifier Training
    batch_size = 256
    epochs = 100
    
    print("\n\nCLASSIFICATION TRAINING: ", ident)
    sc, curr_acc, epoch_data = model_train(model, x2t_train, yt_train, batch_size, epochs, x2t_valid, yt_valid, x2t_test,
                                           yt_test)

    epoch_data.history['user_id'] = idents[ident]
    user_epoch_data[idents[ident]] = epoch_data.history

    jsonformat = json.dumps(str(user_epoch_data))
    f = open("./AEOutput/compl2_2_NC_kernel_200_pool_4_filter_64_strides_4_lstm_256.json", "w")
    f.write(jsonformat)
    f.close()


# In[ ]:





# ### Sparse Autoencoders
# TODO - more or less copy undercomplete autoencoders, but add an extra parameter.

# In[ ]:





# ### Deep Autoencoders 

# #### Deep Autoencoder - Normalized Input Data
# from https://blog.keras.io/building-autoencoders-in-keras.html  
# Just testing, still work in progress.

# Build the autoencoder:

# In[80]:


# From https://www.tensorflow.org/guide/keras/functional#use_the_same_graph_of_layers_to_define_multiple_models
encoder_input = keras.Input(shape=x_train[0].shape, name="normalized_signal")
x = layers.Dropout(0.1, name="dropout", autocast=False)(encoder_input)
x = layers.Dense(512, activation="relu", name="dense_enc_1", autocast=False)(x)
x = layers.Dense(256, activation="relu", name="dense_enc_2")(x)
encoder_output = layers.Dense(64, activation="relu", name="encoded_signal")(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")

x = layers.Dense(256, activation="sigmoid", name="dense_dec_1")(encoder_output)
x = layers.Dense(512, activation="sigmoid", name="dense_dec_2")(x)
decoder_output = layers.Dense(x_train.shape[1], activation="sigmoid", name="reconstructed_signal")(x)
autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")


# Print the summary of the autoencoder:

# In[81]:


autoencoder.summary()
keras.utils.plot_model(autoencoder, "./AEOutput/deep_ae.png", show_shapes=True)


# Compile and train the model:

# In[82]:


# autoencoder.compile(optimizer='adam', loss='MSE')
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
batch_size = 256
epochs = 500

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

# sc, curr_acc, epoch_data = model_train(autoencoder, x_train, x_train, batch_size, epochs, x_valid, x_valid, x_test, x_test)


# In[83]:


x_pred = autoencoder.predict(x_train)


#plot n samples to compare
n = 33

#plot the original values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_train[i])
    
#plot the reconstructed values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_pred[i])


# ### Sequence-To-Sequence Autoencoders
# from https://blog.keras.io/building-autoencoders-in-keras.html

# #### Sequence-To-Sequence Autoencoder - Normalized Input Data
# Just testing, still work in progress.  
# Have to consider a few different layers: TimeDistributed, RepeatVector. And rethink the topology.

# In[84]:


# Convolution
kernel_size = 200
filters = 64
strides = 4
# pooling
pool_size = 4
# LSTM
lstm_output_size = 1290

# x_train = x_train.reshape(-1, x_train[0].shape[0], 1)
# x_valid = x_valid.reshape(-1, x_valid[0].shape[0], 1)
# x_test = x_test.reshape(-1, x_test[0].shape[0], 1)

model = Sequential()
model.add(layers.Reshape((x_test[0].shape[0],1),input_shape=x_train[0].shape))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=strides))

model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=strides))
model.add(MaxPooling1D(pool_size=pool_size))

# model.add(Dropout(0.25))
model.add(LSTM(lstm_output_size))
model.add(Activation('sigmoid'))

autoencoder = model

# input_dim=1290
# latent_dim=1290
# timesteps=1

# inputs = Input(shape=(timesteps, 1290))
# encoded = LSTM(latent_dim)(inputs)

# decoded = layers.RepeatVector(timesteps)(encoded)
# decoded = LSTM(input_dim, return_sequences=True)(decoded)

# autoencoder = Model(inputs, decoded)
# encoder = Model(inputs, encoded)


# In[85]:


y = autoencoder(x_train)
autoencoder.summary()


# In[ ]:





# In[86]:


# autoencoder.compile(optimizer='adam', loss='MSE')
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
batch_size = 256
epochs = 50

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

# sc, curr_acc, epoch_data = model_train(autoencoder, x_train, x_train, batch_size, epochs, x_valid, x_valid, x_test, x_test)


# In[87]:


x_pred = autoencoder.predict(x_train)


#plot n samples to compare
n = 33

#plot the original values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_train[i])
    
#plot the reconstructed values
plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(n/5, 6, i+1)
    plt.plot(x_pred[i])


# In[ ]:





# In[ ]:





# In[ ]:




