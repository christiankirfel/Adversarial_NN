#Adding arguments for the queue 
# coding: utf-8

# # 1. Set up environment

# ## 1.1 Select Theano as backend for Keras
# Tensorflow also works with ATLBONNTW-81

# In[5]:

import sys

import matplotlib
matplotlib.use('Agg')

#Set up the name for the output files
file_extension = sys.argv[1] + '_' + str(int(sys.argv[2]) +1) + '_' + sys.argv[3]
output_path = './figures_' + file_extension + '/'

#!/usr/local/bin python3

from os import environ
import os
environ['KERAS_BACKEND'] = 'theano'
#environ['KERAS_BACKEND'] = 'tensorflow'

# Set architecture of system (AVX instruction set is not supported on SWAN)
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'

#Create the output file

if not os.path.exists(output_path):
    os.makedirs(output_path)


# ## 1.2 Read variables and options

# In[6]:


# # Process command line arguments
# if not len(sys.argv)==2:
#     print('Usage: python AdverNet.py <region>')
#     sys.exit()
# region = str(sys.argv[1])

#Choose region

region = "1j1b"
#region = "tZ"

#Choose variable set

#var_region = "3j1b"
#var_region = "3j1b_basic"

# In[7]:


#with open('Variables_'+region+'.txt','r') as varfile:
#with open('Variables_'+var_region+'.txt','r') as varfile:
with open('Variables_'+sys.argv[1]+'.txt','r') as varfile:
    variableList = varfile.read().splitlines() 

def ReadOptions(region):
    with open('KerasOptions_'+region+'.txt','r') as infile:
        optionsList = infile.read().splitlines()
    OptionDict = dict()
    for options in optionsList:
		# definition of a comment
        if options.startswith('#'): continue
        templist = options.split(' ')
        if len(templist) == 2:
            OptionDict[templist[0]] = templist[1]
        else:
            OptionDict[templist[0]] = templist[1:]
    return OptionDict
    # a dictionary of options is returned

Options = ReadOptions(region)

print (variableList, Options['EventWeight'])
print (Options)



# # 2. Load data

# ## 2.1 Read ROOT file
# Two ways to read data:
# https://indico.fnal.gov/event/13497/session/1/material/slides/0?contribId=47 page 12




from root_numpy import root2array, tree2array
import ROOT
from numpy import *

#Get TTree from pyROOT then convert to numpy array
#file = ROOT.TFile('data/'+str(Options['File'])+'_'+region+'_nominal.root')
#file = ROOT.TFile('lustre/lustre/'+str(Options['File'])+'_'+region+'_nominal.root')
#single lepton
file = ROOT.TFile('reprocess_3j1b_nominal_kirfel_testedit.root')
#file = ROOT.TFile('MC_training/reprocessNB_3j1b_data_nominal.root')
#filelist = [ ROOT.TFile('~/internship/MC_training/' + file) for file in os.listdir('MC_training') ]
#file.ls()


# In[ ]:


''' *array* is [eventVariables, EventWeight]; *event* is [eventVariables]; *weight* is [EventWeight]'''
tree__signal = []
tree__background = []
event__signal = []
event__background = []
weight__signal = []
weight__background = []
array__signal = []
array__background = []

name__signal = Options['SignalTree'] #wt_DR_nominal wt_DS
name__background = Options['BackgroundTree'] #tt_nominal tt_radHi
#Need to add lines to make sure I can use may files
print name__signal
#for filename in input:


  
for name in name__signal:
    if file.Get(name) != None:
       print 'name', name
       tree__signal.append(file.Get(name))
       event__signal.append(tree2array(tree__signal[-1], branches=variableList, selection='1'))
       weight__signal.append(tree2array(tree__signal[-1], branches=[ Options['EventWeight'] ], selection='1'))
#     weight__signal.append(tree2array(tree__signal[-1], branches="EventWeight", selection='1'))
       array__signal.append([list(elem) for elem in zip(event__signal[-1], weight__signal[-1])])
for name in name__background:
    if file.Get(name) != None:
       tree__background.append(file.Get(name))
       event__background.append(tree2array(tree__background[-1], branches=variableList, selection='1'))
       weight__background.append(tree2array(tree__background[-1], branches=[ Options['EventWeight'] ], selection='1'))
       array__background.append([list(elem) for elem in zip(event__background[-1], weight__background[-1])])
    
    #above reads in

if bool(int(Options['UseWeight'])) is False:
    for weight in weight__signal:
        weight[:] = 1
    for weight in weight__background:
        weight[:] = 1
    print ('EventWeight set to 1')
    
    #set entire list to 1


# ## 2.2 Split into training and test sets

# Construct **train\_\_sample\_nominal**, **test\_\_sample\_nominal** 
# and their coresponding score **targettrain\_\_sample\_nominal**
# **targettest\_\_sample\_nominal**

# In[ ]:


'''using Options['TrainFraction'] to control fractions of training and test samples'''
import numpy as np
from copy import deepcopy
from sklearn.cross_validation import train_test_split

def weight_ratio(weight__signal, weight__background):
    total_weight__signal = total_weight__background = 0
    for weight in weight__signal:
        total_weight__signal += sum(j[0] for j in [list(i) for i in weight])
    for weight in weight__background:
        total_weight__background += sum(j[0] for j in [list(i) for i in weight])
    return total_weight__signal / total_weight__background

ratiotWtt = weight_ratio(weight__signal, weight__background)
# ratiotWtt = sum(j[0] for j in [list(i) for i in weight__signal[0]]) / sum(j[0] for j in [list(i) for i in weight__background[0]])

train_array__signal = []
test_array__signal = []
train_array__background = []
test_array__background = []

''' *array* is [eventVariables, EventWeight]; *event* is [eventVariables]; *weight* is [EventWeight] '''
''' Construct train and test for wt_DR, tt, wt_DS '''
for array in array__signal:
    train_array, test_array = train_test_split(array, train_size=float(Options['TrainFraction']), test_size=1-float(Options['TrainFraction']), random_state = 1)
    train_array__signal.append(deepcopy(train_array))
    test_array__signal.append(deepcopy(test_array))

for array in array__background:
    train_array, test_array = train_test_split(array, train_size=float(Options['TrainFraction']), test_size=1-float(Options['TrainFraction']), random_state = 1)
    train_array__background.append(deepcopy(train_array))
    test_array__background.append(deepcopy(test_array))



# In[ ]:


train_event__signal = []
train_weight__signal = []
train_event__background = []
train_weight__background = []
test_event__signal = []
test_weight__signal = []
test_event__background = []
test_weight__background = []

for train_array in train_array__signal:
    train_event__signal.append([list(i[0]) for i in train_array])
    train_weight__signal.append([j[0]/ratiotWtt for j in [list(i[1]) for i in train_array]])

for train_array in train_array__background:
    train_event__background.append([list(i[0]) for i in train_array])
    train_weight__background.append([j[0] for j in [list(i[1]) for i in train_array]])

for test_array in test_array__signal:
    test_event__signal.append([list(i[0]) for i in test_array])
    test_weight__signal.append([j[0]/ratiotWtt for j in [list(i[1]) for i in test_array]])

for test_array in test_array__background:
    test_event__background.append([list(i[0]) for i in test_array])
    test_weight__background.append([j[0] for j in [list(i[1]) for i in test_array]])


''' Construct target for train and test for wt_DR, tt, wt_DS
    wt = 1; tt = 0 '''
train_target__signal = []
test_target__signal = []
train_target__background = []
test_target__background = []

for train_array in train_array__signal:
    train_target__signal.append(np.arange(len(train_array)))
    train_target__signal[-1][:] = 1
for test_array in test_array__signal:
    test_target__signal.append(np.arange(len(test_array)))
    test_target__signal[-1][:] = 1
for train_array in train_array__background:
    train_target__background.append(np.arange(len(train_array)))
    train_target__background[-1][:] = 0
for test_array in test_array__background:
    test_target__background.append(np.arange(len(test_array)))
    test_target__background[-1][:] = 0


''' Construct systematics for train and test for wt_DR, tt, wt_DS
    wt_DR = tt = 0; wt_DS = 1 '''
train_systematics__signal = []
test_systematics__signal = []
train_systematics__background = []
test_systematics__background = []

for train_array in train_array__signal:
    train_systematics__signal.append(np.arange(len(train_array)))
    train_systematics__signal[-1][:] = 0 if len(train_systematics__signal)==1 else 1
for test_array in test_array__signal:
    test_systematics__signal.append(np.arange(len(test_array)))
    test_systematics__signal[-1][:] = 0 if len(test_systematics__signal)==1 else 1
for train_array in train_array__background:
    train_systematics__background.append(np.arange(len(train_array)))
    train_systematics__background[-1][:] = 0 if len(train_systematics__background)==1 else 1
for test_array in test_array__background:
    test_systematics__background.append(np.arange(len(test_array)))
    test_systematics__background[-1][:] = 0 if len(test_systematics__background)==1 else 1


for i in range(len(array__signal)):
    assert (len(train_array__signal[i])+len(test_array__signal[i]) == len(array__signal[i]))
for i in range(len(array__background)):
    assert (len(train_array__background[i])+len(test_array__background[i]) == len(array__background[i]))


print ('Training sample wt_DR_nominal: ', len(train_event__signal[0]), '\n',
       '                tt_nominal:   ', len(train_event__background[0]))
for i in range(1, len(train_event__signal)):
    print ('                 wt syst', i, ':   ', len(train_event__signal[i]))
for i in range(1, len(train_event__background)):
    print ('                 tt syst', i, ':   ', len(train_event__background[i]))
print('              total nominal:   ', len(train_event__signal[0]) + len(train_event__background[0]))
print ('Test sample wt_DR_nominal: ', len(test_event__signal[0]), '\n',
       '           tt_nominal:    ', len(test_event__background[0]))
for i in range(1, len(test_event__signal)):
    print('            wt syst', i, ':    ', len(test_event__signal[i]))
for i in range(1, len(test_event__background)):
    print('            tt_syst', i, ':    ', len(test_event__background[i]))
print('         total nominal:    ', len(test_event__signal[0]) + len(test_event__background[0]))

                


# Merge signal and backgrounds to **train(test)\_\_event(weight)\_\_nominal** 
# and their coresponding score **targettrain(test)\_\_nominal**
# **systematicstrain(test)\ `python': corrupted size vs. prev_size: 0x0000000008aecb70 ***
#_\_nominal**

# In[ ]:


''' Construct sample, EventWeight, target, systematics of train and test
    mixing parts of wt_DR, tt, wt_DS '''

train_event__list = []
for train_event in train_event__signal:
    train_event__list.append(train_event)
for train_event in train_event__background:
    train_event__list.append(train_event)
train_event = np.vstack(train_event__list)

test_event__list = []
for test_event in test_event__signal:
    test_event__list.append(test_event)
for test_event in test_event__background:
    test_event__list.append(test_event)
test_event = np.vstack(test_event__list)

train_weight__list = []
for train_weight in train_weight__signal:
    train_weight__list.append(train_weight)
for train_weight in train_weight__background:
    train_weight__list.append(train_weight)
train_weight = np.concatenate(train_weight__list)

test_weight__list = []
for test_weight in test_weight__signal:
    test_weight__list.append(test_weight)
for test_weight in test_weight__background:
    test_weight__list.append(test_weight)
test_weight = np.concatenate(test_weight__list)

train_target__list = []
for train_target in train_target__signal:
    train_target__list.append(train_target)
for train_target in train_target__background:
    train_target__list.append(train_target)
train_target = np.concatenate(train_target__list)

test_target__list = []
for test_target in test_target__signal:
    test_target__list.append(test_target)
for test_target in test_target__background:
    test_target__list.append(test_target)
test_target = np.concatenate(test_target__list)


train_systematics__list = []
for train_systematics in train_systematics__signal:
    train_systematics__list.append(train_systematics)
for train_systematics in train_systematics__background:
    train_systematics__list.append(train_systematics)
train_systematics = np.concatenate(train_systematics__list)

test_systematics__list = []
for test_systematics in test_systematics__signal:
    test_systematics__list.append(test_systematics)
for test_systematics in test_systematics__background:
    test_systematics__list.append(test_systematics)
test_systematics = np.concatenate(test_systematics__list)


# In[ ]:


''' Data conversion of sample '''
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
train_event_transfered = scaler.fit_transform(train_event)
print type(scaler)

outfolder = 'results/' + Options['Output'] + '/'
#store the content
print 'before open'
print outfolder + Options['Pkl'] + '.pkl'
with open(Options['Pkl'] + '.pkl', 'wb') as handle:
    pickle.dump(scaler, handle)
#load the content
#scaler = pickle.load(outfolder + open(Options['Pkl']+'.pkl', 'rb' ) )
scaler = pickle.load(open(Options['Pkl']+'.pkl', 'rb' ) )
test_event_transfered = scaler.transform(test_event)

assert (train_event_transfered.shape[1] == len(variableList))


# # 3. Simple networks
#could coment out

# ## 3.1 Build networks by Keras

# In[ ]:
#CARE PLOT_MODEL COMMENDTED OUT

#import keras.backend as K
#from keras.layers import Input, Dense
#from keras.models import Model
#from keras.optimizers import SGD
##from keras.utils.vis_utils import plot_model

## 3 hiddend layers
#simple_inputs = Input(shape=(train_event_transfered.shape[1],), name='Net_input')
#simple_Dx = Dense(32, activation="relu", name='Net_layer1')(simple_inputs)
#simple_Dx = Dense(32, activation="relu", name='Net_layer2')(simple_Dx)
#simple_Dx = Dense(32, activation="relu", name='Net_layer3')(simple_Dx)
#simple_Dx = Dense(32, activation="relu", name='Net_layer4')(simple_Dx)
#simple_Dx = Dense(32, activation="relu", name='Net_layer5')(simple_Dx)
#simple_Dx = Dense(1, activation="sigmoid", name='Net_output')(simple_Dx)
#simple_D = Model(inputs=[simple_inputs], outputs=[simple_Dx], name='Net_model')
#simple_D.compile(loss="binary_crossentropy", optimizer="adam")
#simple_D.summary()
##plot_model(simple_D, to_file='png/simple_D.png')


## ## 3.2 Train

## In[ ]:


#''' Train on train_event with target train_target, using train_weight as EventWeight '''
#simple_D.fit(train_event_transfered, train_target, sample_weight=train_weight, epochs=int(Options['SimpleTrainEpochs']))


## ## 3.3 Test

## In[ ]:


#''' Apply training results to test sample; and training sample for checking '''
#from sklearn.metrics import roc_auc_score
#predicttest__simple_D = simple_D.predict(test_event_transfered)
#predicttrain__simple_D = simple_D.predict(train_event_transfered)


## ## 3.4 Calculate and plot ROC

## In[ ]:


#from sklearn.metrics import roc_curve, auc

#print('Traing ROC: ', roc_auc_score(train_target, predicttrain__simple_D))
#print('Test ROC:   ', roc_auc_score(test_target, predicttest__simple_D))


## In[ ]:


#''' Plot ROC '''
#import matplotlib.pyplot as plt

#train__false_positive_rate, train__true_positive_rate, train__thresholds = roc_curve(train_target, predicttrain__simple_D)
#test__false_positive_rate, test__true_positive_rate, test__thresholds = roc_curve(test_target, predicttest__simple_D)
#train__roc_auc = auc(train__false_positive_rate, train__true_positive_rate)
#test__roc_auc = auc(test__false_positive_rate, test__true_positive_rate)


#plt.title('Receiver Operating Characteristic')
#plt.plot(train__false_positive_rate, train__true_positive_rate, 'g--', label='Train AUC = %0.2f'% train__roc_auc)
#plt.plot(test__false_positive_rate, test__true_positive_rate, 'b', label='Test AUC = %0.2f'% test__roc_auc)

#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.,1.])
#plt.ylim([-0.,1.])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.gcf().savefig("simple_network_ROC.png")
#plt.show()
#plt.gcf().clear()

##print "First plot I found"


## ## 3.5 Plot traing and test distributions

## In[ ]:


#''' Plot NN output '''
#xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
#SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

#plt.subplot(1, 2, 1)
#plt.hist(predicttrain__simple_D[train_target == 1], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' training')
#plt.hist(predicttrain__simple_D[train_target == 0], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' training')
#plt.hist(predicttest__simple_D[test_target == 1],   range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' test', linestyle='dashed')
#plt.hist(predicttest__simple_D[test_target == 0],   range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' test', linestyle='dashed')
#plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
#plt.legend()
#plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
#plt.title('Absolute')

#plt.subplot(1, 2, 2)
#plt.hist(predicttrain__simple_D[train_target == 1], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' training')
#plt.hist(predicttrain__simple_D[train_target == 0], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' training')
#plt.hist(predicttest__simple_D[test_target == 1],   range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' test', linestyle='dashed')
#plt.hist(predicttest__simple_D[test_target == 0],   range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' test', linestyle='dashed')
#plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
#plt.legend()
#plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
#plt.title('Normalised')
#plt.gcf().savefig('simple_network_NN.png')
#plt.show()

#plt.gcf().clear()


#print ('Test sample wt_DR_nominal: ', len(predicttest__simple_D[test_target == 1]), '\n',
       #'           tt_nominal:    ', len(predicttest__simple_D[test_target == 0]), '\n',
       #'           total:         ', len(predicttest__simple_D))


## In[ ]:


#xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
#SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

#plt.subplot(1, 2, 1)
#plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' norm')
#plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' norm')
#plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' syst', linestyle='dashed')
#plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' syst', linestyle='dashed')
#plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
#plt.legend() 

#plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
#plt.title('Absolute')

#plt.subplot(1, 2, 2)
#plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' norm')
#plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' norm')
#plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' syst', linestyle='dashed')
#plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' syst', linestyle='dashed')
#plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
#plt.legend()
#plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
#plt.title('Normalised')
#plt.gcf().savefig('simple_network_NN2.png')
#plt.show()
#plt.gcf().clear()


#print ('Test sample wt_DR_nominal: ', len(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 0)]), '\n',
       #'            wt_DS_nominal: ', len(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 1)]), '\n',
       #'               tt_nominal: ', len(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 0)]), '\n',
       #'           tt systematics: ', len(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 1)])
      #)


# ## 4. Adversarial networks

# ## 4.1 Build networks by Keras

# In[ ]:


import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

layercount = 0

# 3 hiddend layers
ANN_inputs = Input(shape=(train_event_transfered.shape[1],), name='Net_f_input')
for layercount in(0, sys.argv[2]):
	ANN_Dx = Dense(int(sys.argv[3]), activation="relu", name='Net_f_layer_%d' % int(layercount))(ANN_inputs)

#ANN_Dx = Dense(32, activation="relu", name='Net_f_layer2')(ANN_Dx)
#ANN_Dx = Dense(32, activation="relu", name='Net_f_layer3')(ANN_Dx)
#ANN_Dx = Dense(32, activation="relu", name='Net_f_layer4')(ANN_Dx)
#ANN_Dx = Dense(32, activation="relu", name='Net_f_layer5')(ANN_Dx)
ANN_Dx = Dense(1, activation="sigmoid", name='Net_f_output')(ANN_Dx)
ANN_D = Model(inputs=[ANN_inputs], outputs=[ANN_Dx], name='Net_f_model')

layercount = 0

# ANN_Rx = ANN_Dx
ANN_Rx = ANN_D(ANN_inputs)
for layercount in(0, sys.argv[2]):
	ANN_Rx = Dense(int(sys.argv[3]), activation="relu", name='Net_r_layer_%d' % int(layercount))(ANN_Rx)

ANN_Rx = Dense(1, activation="sigmoid", name='Net_r_output')(ANN_Rx)
ANN_R = Model(inputs=[ANN_inputs], outputs=[ANN_Rx], name='Net_r_model')

lam = -10  # control the trade-off between classification performance and independence

# Loss definitions

def make_loss_ANN_D(c):
    def loss_ANN_D(z_true, z_pred):
        return c * K.binary_crossentropy(z_true, z_pred)
    return loss_ANN_D

def make_loss_ANN_R(c):
    def loss_ANN_R(z_true, z_pred):
        return c * K.binary_crossentropy(z_true, z_pred)
    return loss_ANN_R

def make_trainable(network, flag):
    network.trainable = flag
    network.compile
    for l in network.layers:
        l.trainable = flag

opt_ANN_D = SGD()
ANN_D.compile(loss=[make_loss_ANN_D(c=1.0)], optimizer=opt_ANN_D)
# ANN_D.summary()
#plot_model(ANN_D, to_file='png/ANN_D.png')
#plot_model(ANN_R, to_file='png/ANN_R.png')

opt_ANN_DRf = SGD(momentum=0.3)
ANN_DRf = Model(inputs=[ANN_inputs], outputs=[ANN_D(ANN_inputs), ANN_R(ANN_inputs)])
make_trainable(ANN_R, False)
make_trainable(ANN_D, True)
ANN_DRf.compile(loss=[make_loss_ANN_D(c=1.0), make_loss_ANN_R(c=lam)], optimizer=opt_ANN_DRf)
#plot_model(ANN_DRf, to_file='png/ANN_DRf.png')

opt_ANN_DfR = SGD(momentum=0.2)
ANN_DfR = Model(inputs=[ANN_inputs], outputs=[ANN_R(ANN_inputs)])
make_trainable(ANN_R, True)
make_trainable(ANN_D, False)
ANN_DfR.compile(loss=[make_loss_ANN_R(c=1.0)], optimizer=opt_ANN_DfR)
#plot_model(ANN_DfR, to_file='png/ANN_DfR.png')


# ## 4.2 Pretrain ANN_D

# In[ ]:


''' Pretraining of ANN_D on train_event_transfered with target train_target, using train_weight as EventWeight '''
make_trainable(ANN_R, False)
make_trainable(ANN_D, True)
# ANN_D.compile(loss=ANN_D.loss, optimizer=ANN_D.optimizer)
ANN_D.fit(train_event_transfered, train_target, sample_weight=train_weight, epochs=int(Options['PreTrainEpochs']))


# ## 4.3 ANN_D only results

# In[ ]:


# ''' Define a function to plot losses '''

# from IPython import display

# def plot_loss(i, loss):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())

#     ax1 = plt.subplot(311)   
#     values = np.array(loss["L_D"])
#     print(values)
#     plt.plot(range(len(values)), values, label=r"$L_D$", color="blue")
#     plt.legend(loc="upper right")
#     plt.grid()
    
#     plt.show()


# In[ ]:


# loss = {"L_D": []}

# for i in range(61):

#     l = ANN_D.evaluate(test_event_transfered, test_target, sample_weight=test_weight, verbose=0)
#     print(l)
#     loss["L_D"].append(l)
#     plot_loss(i, loss)
#     ANN_D.fit(train_event_transfered, train_target, sample_weight=train_weight, epochs=1, verbose=1)


# ### 4.3.1 ANN_D only ROC

# In[ ]:


from sklearn.metrics import roc_curve, auc

''' Apply training results to test sample; and training sample for checking '''
predicttrain__ANN_D = ANN_D.predict(train_event_transfered)
predicttest__ANN_D = ANN_D.predict(test_event_transfered)

''' Plot ROC and NN outpout '''



from sklearn.metrics import roc_auc_score
print('Traing ROC: ', roc_auc_score(train_target, predicttrain__ANN_D))
print('Test ROC:   ', roc_auc_score(test_target, predicttest__ANN_D))

import matplotlib.pyplot as plt

train__false_positive_rate, train__true_positive_rate, train__thresholds = roc_curve(train_target, predicttrain__ANN_D)
test__false_positive_rate, test__true_positive_rate, test__thresholds = roc_curve(test_target, predicttest__ANN_D)
train__roc_auc = auc(train__false_positive_rate, train__true_positive_rate)
test__roc_auc = auc(test__false_positive_rate, test__true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(train__false_positive_rate, train__true_positive_rate, 'g--', label='Train AUC = %0.2f'% train__roc_auc)
plt.plot(test__false_positive_rate, test__true_positive_rate, 'b', label='Test AUC = %0.2f'% test__roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.,1.])
plt.ylim([-0.,1.])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.gcf().savefig(output_path + 'ANN_D_ROC_' + file_extension + '.png')
#plt.show()
plt.gcf().clear()


# ### 4.3.2 ANN_D only overtaining

# In[ ]:


xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

plt.subplot(1, 2, 1)
plt.hist(predicttrain__ANN_D[train_target == 1], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' training')
plt.hist(predicttrain__ANN_D[train_target == 0], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' training')
plt.hist(predicttest__ANN_D[test_target == 1],   range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' test', linestyle='dashed')
plt.hist(predicttest__ANN_D[test_target == 0],   range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' test', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN_D response', horizontalalignment='left', fontsize='large')
plt.title('Absolute')

plt.subplot(1, 2, 2)
plt.hist(predicttrain__ANN_D[train_target == 1], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' training')
plt.hist(predicttrain__ANN_D[train_target == 0], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' training')
plt.hist(predicttest__ANN_D[test_target == 1],   range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' test', linestyle='dashed')
plt.hist(predicttest__ANN_D[test_target == 0],   range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' test', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN_D response', horizontalalignment='left', fontsize='large')
plt.title('Normalised')
#attention---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#plt.bak-modelexpshow()
plt.gcf().savefig(output_path + 'ANN_D_NN_' + file_extension + '.png')
plt.gcf().clear()

print ('Test sample wt_DR_nominal: ', len(predicttest__ANN_D[test_target == 1]), '\n',
       '           tt_nominal:    ', len(predicttest__ANN_D[test_target == 0]), '\n',
       '           total:         ', len(predicttest__ANN_D))


# ### 4.3.3 ANN_D only syst deviations

# In[ ]:


xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

plt.subplot(1, 2, 1)
plt.hist(predicttest__ANN_D[logical_and(test_target == 1, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' norm')
plt.hist(predicttest__ANN_D[logical_and(test_target == 0, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' norm')
plt.hist(predicttest__ANN_D[logical_and(test_target == 1, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' syst', linestyle='dashed', )
plt.hist(predicttest__ANN_D[logical_and(test_target == 0, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' syst'
         , linestyle='dashed', )
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN_D response', horizontalalignment='left', fontsize='large')
plt.title('Absolute')

plt.subplot(1, 2, 2)
plt.hist(predicttest__ANN_D[logical_and(test_target == 1, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' norm')
plt.hist(predicttest__ANN_D[logical_and(test_target == 0, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' norm')
plt.hist(predicttest__ANN_D[logical_and(test_target == 1, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' syst', linestyle='dashed', )
plt.hist(predicttest__ANN_D[logical_and(test_target == 0, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' syst', linestyle='dashed', )
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN_D response', horizontalalignment='left', fontsize='large')
plt.title('Normalised')
plt.gcf().savefig(output_path + 'ANN_D_syst_dev_' + file_extension + '.png')
#plt.show()
plt.gcf().clear()

print ('Test sample wt_DR_nominal: ', len(predicttest__ANN_D[logical_and(test_target == 1, test_systematics == 0)]), '\n',
       '            wt_DS_nominal: ', len(predicttest__ANN_D[logical_and(test_target == 1, test_systematics == 1)]), '\n',
       '               tt_nominal: ', len(predicttest__ANN_D[logical_and(test_target == 0, test_systematics == 0)]), '\n',
       '           tt systematics: ', len(predicttest__ANN_D[logical_and(test_target == 0, test_systematics == 1)])
      )



# ## 4.4 Pretrain ANN_R

# In[ ]:


''' Pretraining of ANN_R on train_event_transfered with target train_systematics, using train_weight as EventWeight '''
make_trainable(ANN_R, True)
make_trainable(ANN_D, False)
# ANN_DfR.compile(loss=ANN_DfR.loss, optimizer=ANN_DfR.optimizer)
ANN_DfR.fit(train_event_transfered, train_systematics, sample_weight=train_weight, epochs=int(Options['PreTrainEpochs']))


# ## 4.5 Train adversarial networks

# In[ ]:


''' Define a function to plot losses '''

from IPython import display
import matplotlib.pyplot as plt


def plot_losses(i, losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())

    ax1 = plt.subplot(311)   
    values = np.array(losses["L_f"])
    plt.plot(range(len(values)), values, label=r"$loss_D$", color="blue")
    plt.legend(loc="upper right")
    plt.grid()
    
    ax2 = plt.subplot(312, sharex=ax1) 
    values = np.array(losses["L_r"]) / lam
    plt.plot(range(len(values)), values, label=r"$loss_R$", color="green")
    plt.legend(loc="upper right")
    plt.grid()
    
    ax3 = plt.subplot(313, sharex=ax1)
    values = np.array(losses["L_f - L_r"])
    plt.plot(range(len(values)), values, label=r"$loss_D "+str(lam)+r"*loss_R$", color="red")  
    plt.legend(loc="upper right")
    plt.grid()
    plt.gcf().savefig(output_path + 'losses_' + file_extension + '_' + str(i/5) + '.png')
    plt.gcf().clear()
    #plt.show()



# ### 4.5.1 Adversarial networks losses

# In[ ]:


# ReadFiles()

losses = {"L_f": [], "L_r": [], "L_f - L_r": []}
ANN_DRf.metrics_names
batch_size = 128
for i in range(int(Options['Iterations'])):
# for i in range(2):

    l = ANN_DRf.evaluate(test_event_transfered, [test_target, test_systematics], sample_weight=[test_weight, test_weight], verbose=0)   
    
    losses["L_f"].append(l[1][None][0])
    losses["L_r"].append(l[2][None][0])
    losses["L_f - L_r"].append(l[0][None][0])
    print('zhangrui', losses["L_f"][-1], losses["L_r"][-1]/lam, '*', lam, losses["L_f - L_r"][-1])
    
    print(losses)


 
    if i % 5 == 0:
        plot_losses(i, losses)

    #Fit ANN_D
    ''' Problem with train_on_batch is EventWeight does not shuffle with events
        Memory can handle all events training
        - need to check epochs '''
    make_trainable(ANN_R, False)
    make_trainable(ANN_D, True)

    indices = np.random.permutation(len(train_event_transfered))[1:]
    if i % 3 == 0:
        ANN_DRf.fit(train_event_transfered[indices], [train_target[indices], train_systematics[indices]], sample_weight=[train_weight[indices], train_weight[indices]], 
                epochs=int(Options['AdTrainEpochs']), verbose=1)
    else:
        ANN_DRf.train_on_batch(train_event_transfered[indices], [train_target[indices], train_systematics[indices]], sample_weight=[train_weight[indices], train_weight[indices]])

        #     ANN_DRf.fit(train_event_transfered, [train_target, train_systematics], sample_weight=[train_weight, train_weight], 
#                 epochs=int(Options['AdTrainEpochs']), verbose=1)


    # Fit ANN_R
    make_trainable(ANN_R, True)
    make_trainable(ANN_D, False)
#     ANN_DfR.compile(loss=ANN_DfR.loss, optimizer=ANN_DfR.optimizer)
#     ANN_DfR.train_on_batch(train_event_transfered, train_systematics, sample_weight=train_weight)
    ANN_DfR.fit(train_event_transfered, train_systematics, sample_weight=train_weight, batch_size=batch_size, epochs=1)



# ### 4.5.2 Save model to disk
# https://stackoverflow.com/questions/29788047/keep-tfidf-result-for-predicting-new-content-using-scikit-for-python
# - If you want to store features list for testing data for use in future, you can do this:
#       tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# 
# - store the content.
#       with open("x_result.pkl", 'wb') as handle:
#           pickle.dump(tfidf, handle)
# - load the content
#       tfidf = pickle.load(open("x_result.pkl", "rb" ) )

# In[ ]:

print "Are we still going?"

import os
outfolder = 'results/' + Options['Output'] + '/'
#This comes too late ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#os.mkdir(outfolder)
#Need to give strings as options
# serialize model to JSON
model_json = ANN_D.to_json()
with open(outfolder + Options['Adresult'] + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ANN_D.save_weights(outfolder + Options['Adresult'] + ".h5")
print("Saved "+ outfolder + Options['Adresult'] + " to disk")


# ### 4.5.3 Adversarial networks ROC

# In[ ]:


predicttest__ANN = ANN_D.predict(test_event_transfered)
predicttrain__ANN = ANN_D.predict(train_event_transfered)

from sklearn.metrics import roc_curve, auc

print('Traing ROC: ', roc_auc_score(train_target, predicttrain__ANN))
print('Test ROC:   ', roc_auc_score(test_target, predicttest__ANN))


train__false_positive_rate, train__true_positive_rate, train__thresholds = roc_curve(train_target, predicttrain__ANN)
test__false_positive_rate, test__true_positive_rate, test__thresholds = roc_curve(test_target, predicttest__ANN)
train__roc_auc = auc(train__false_positive_rate, train__true_positive_rate)
test__roc_auc = auc(test__false_positive_rate, test__true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(train__false_positive_rate, train__true_positive_rate, 'g--', label='train AUC = %0.2f'% train__roc_auc)
plt.plot(test__false_positive_rate, test__true_positive_rate, 'b', label='test AUC = %0.2f'% test__roc_auc)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.,1.])
plt.ylim([-0.,1.])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.gcf().savefig(output_path + 'ANN_ROC_' + file_extension + '.png')
#plt.show()


# ### 4.5.4 Adversarial networks overtraining

# In[ ]:


xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

plt.subplot(1, 2, 1)
plt.hist(predicttrain__ANN[train_target == 1], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' training')
plt.hist(predicttrain__ANN[train_target == 0], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' training')
plt.hist(predicttest__ANN[test_target == 1],   range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' test', linestyle='dashed')
plt.hist(predicttest__ANN[test_target == 0],   range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' test', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN response', horizontalalignment='left', fontsize='large')
plt.title('Absolute')

plt.subplot(1, 2, 2)
plt.hist(predicttrain__ANN[train_target == 1], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' training')
plt.hist(predicttrain__ANN[train_target == 0], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' training')
plt.hist(predicttest__ANN[test_target == 1],   range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' test', linestyle='dashed')
plt.hist(predicttest__ANN[test_target == 0],   range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' test', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN response', horizontalalignment='left', fontsize='large')
plt.title('Normalised')
plt.gcf().savefig(output_path + 'ANN_NN_' + file_extension + '.png')
#plt.show()

plt.gcf().clear()

print ('Test sample wt_nominal: ', len(predicttest__ANN[test_target == 1]), '\n',
       '           tt_nominal: ', len(predicttest__ANN[test_target == 0]), '\n',
       '           total:     ', len(predicttest__ANN))


# ### 4.5.5 Adversarial networks syst deviations

# In[ ]:


xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

plt.subplot(1, 2, 1)
plt.hist(predicttest__ANN[logical_and(test_target == 1, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' norm')
plt.hist(predicttest__ANN[logical_and(test_target == 0, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' norm')
plt.hist(predicttest__ANN[logical_and(test_target == 1, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=SignalName+' syst', linestyle='dashed')
plt.hist(predicttest__ANN[logical_and(test_target == 0, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' syst', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN response', horizontalalignment='left', fontsize='large')
plt.title('Absolute')

plt.subplot(1, 2, 2)
plt.hist(predicttest__ANN[logical_and(test_target == 1, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' norm')
plt.hist(predicttest__ANN[logical_and(test_target == 0, test_systematics == 0)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' norm')
plt.hist(predicttest__ANN[logical_and(test_target == 1, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=SignalName+' syst', linestyle='dashed')
plt.hist(predicttest__ANN[logical_and(test_target == 0, test_systematics == 1)], range=[xlo, xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' syst', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('ANN response', horizontalalignment='left', fontsize='large')
plt.title('Normalised')
plt.gcf().savefig(output_path + 'ANN_syst_dev_' + file_extension + '.png')
#plt.show()
plt.gcf().clear()

print ('Test sample wt_DR_nominal: ', len(predicttest__ANN[logical_and(test_target == 1, test_systematics == 0)]), '\n',
       '            wt_DS_nominal: ', len(predicttest__ANN[logical_and(test_target == 1, test_systematics == 1)]), '\n',
       '               tt_nominal: ', len(predicttest__ANN[logical_and(test_target == 0, test_systematics == 0)]), '\n',
       '           tt systematics: ', len(predicttest__ANN[logical_and(test_target == 0, test_systematics == 1)])
      )

print "The variable set was Variables_" + sys.argv[1] 
print "The network used " + str(int(sys.argv[2])+1) + " layers with " + sys.argv[3] + " nodes each"
print file_extension





