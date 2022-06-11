#Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import copy
import itertools

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.metrics import Metric

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as mtr

#Custom Imports
sys.path.append('../src')
import ml_functions as ml
import neural_network as nn

#Loading Data
df = pd.read_csv('../dataset/training_set_(50_50).csv', index_col = 0)
df_test = pd.read_csv('../dataset/testing_set_(90_10).csv', index_col = 0)

#Combining All Data
df_all = pd.concat([df, df_test])

#Dropping unnecessary columns
to_drop = ['account_creation_time','account_active_duration','time_between_first_and_last_transaction',
           'gini_coefficient_accounts_received','gini_coefficient_accounts_sent',
           'gini_coefficient_values_received','gini_coefficient_values_sent']

#All Data Drop
df_all.drop(to_drop, axis = 1, inplace = True)

#Data Shuffle
df_all = df_all.sample(frac = 1, random_state = 2022)

#Constructing Test Set
test_n = 5000
test_ponzi_frac = 0.1

df_test = pd.concat([df_all.loc[df_all.ponzi].iloc[:int(test_n*test_ponzi_frac)],
                     df_all.loc[~df_all.ponzi].iloc[:int(test_n*(1-test_ponzi_frac))]])

x_test = df_test.iloc[:,:-1]
y_test = df_test.ponzi.astype(int)

#Constructing Train Set
mask = [x for x in df_all.index if x not in df_test.index]

x_train_base = df_all.loc[mask].iloc[:,:-1]
y_train_base = df_all.ponzi.loc[mask].astype(int)

validation_frac = 0.2
validation_index = int(len(x_train_base)*(1-validation_frac))+1

x_train = x_train_base.iloc[:validation_index,:]
y_train = y_train_base[:validation_index]

x_val = x_train_base.iloc[validation_index:,:]
y_val = y_train_base[validation_index:]

#Scaling Train Data
scaler = PowerTransformer()
#scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_train = tf.convert_to_tensor(x_train, dtype = tf.float64)
y_train = tf.convert_to_tensor(y_train, dtype = tf.float64)

#Scaling Validation Data
x_val = tf.convert_to_tensor(scaler.transform(x_val), dtype = tf.float64)
y_val = tf.convert_to_tensor(y_val, dtype = tf.float64)

#Scaling Test Data
x_test = tf.convert_to_tensor(scaler.transform(x_test), dtype = tf.float64)
y_test = tf.convert_to_tensor(y_test, dtype = tf.float64)

#Plotting Scaled Data
if to_plot:
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (24,8))
    ax[0].boxplot(x_train.numpy(), showfliers = False)
    ax[0].set_title('train')
    ax[1].boxplot(x_val.numpy(), showfliers = False)
    ax[1].set_title('validation')
    ax[2].boxplot(x_test.numpy(), showfliers = False)
    ax[2].set_title('test')
    plt.show()

#Specifying NN Hyperparameters
#Design
layer_types = ['dense', 'dropout'] * 4

n_nodes_per_layer = 512

n_nodes = [n_nodes_per_layer if x == 'dense' else None for x in layer_types]
activations = [tf.nn.leaky_relu if x == 'dense' else None for x in layer_types]
dropout_param = 0.25
optimiser = tf.keras.optimizers.Adam
loss_fn = tf.losses.BinaryCrossentropy()
start_learn_rate = 1E-4


callback_es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 100, verbose = 1)
callback_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 20,
                                                    min_delta = 1E-4, cooldown = 20, min_lr = 1E-6, verbose = 1)

#Training
n_epochs = 1000
batch_size = 32

#Creating NNs
nn7 = nn.NeuralNetwork(x_train, y_train, x_val, y_val,
                       layer_types, n_nodes, activations, dropout_param, optimiser, loss_fn, start_learn_rate,
                       filepath = r'C:\Users\quekh\Desktop\temp\nn7', long_folder = False)

#Training
nn7.model_fit(batch_size, n_epochs, callback_list = [callback_es, callback_rlr], save_checkpoints = True)

#Saving
nn7.save_model()

nn7.m.evaluate(x_test,y_test)

nn7.plot_metrics(0, 200)


def plot_confusion_matrix(cm, classes, normalize=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix', size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)


def plot_roc_curve(y, prob):
    base_fpr, base_tpr, _ = mtr.roc_curve(y, [1 for _ in range(len(y))])
    model_fpr, model_tpr, _ = mtr.roc_curve(y, prob)
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='rf')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC Curves');
    plt.show();


def train_test_measure(y_test, y_pred, y_pred_proba):
    cm = mtr.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['0 - Normal', '1 - Ponzi'])

    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    recall = mtr.recall_score(y_test, y_pred)
    results = (mtr.precision_recall_fscore_support(y_test, y_pred, beta=2))
    print('accuracy:', mtr.accuracy_score(y_test, y_pred))
    #    print('precision:', mtr.precision_score(yte,y_test_pred))
    #    print('recall:', recall)
    print('specificity:', specificity)
    print('precision non-fraud :', results[0][0], 'precision fraud :', results[0][1])
    print('recall non-fraud:', results[1][0], "recall fraud", results[1][1])
    print('f2 non-fraud', results[2][0], "f2 fraud : ", results[2][1])
    print('f1:', mtr.f1_score(y_test, y_pred))
    print('g mean: ', (recall * specificity) ** 0.5)
    print('auc:', mtr.roc_auc_score(y_test, y_pred_proba))
    plot_roc_curve(y_test, y_pred_proba)

# Summary Performance
y_pred_proba = nn7.m.predict(x_test)
y_pred = np.array([1 if y > 0.5 else 0 for y in y_pred_proba])
train_test_measure(y_test, y_pred, y_pred_proba)