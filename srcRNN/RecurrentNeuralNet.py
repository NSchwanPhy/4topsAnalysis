import numpy as np
import PlotService
from Utils import *
from Guard import GuardRNN
from Callbacks import *
from tensorflow.keras import optimizers
import ROOT
import os
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K

def Main(ANNSetup,DataSet,BootStrap=('vali',None)):
    """
    Transforms the DataSet to a keras readable version and selectes the correct layer type

    ANNSetup:      Definition of the hyperparameters of the NN
    DataSet:       Dataset created by the Sampler
    BootStrap:     Bootstrap setup tuble (Sample,random seed)

    Output:        trained model and Aucs for each training step
    """
    
    # Random seed for the NN cacluations
    np.random.seed(5)
    tf.compat.v1.set_random_seed(5)

    train, test, vali = GetSamples(BootStrap,DataSet,ANNSetup.ModelName) # Transform the DataSet into keras readable and aplying bootstrap if needed
    GuardRNN(ANNSetup)                                                   # Protect the setup from typos
    if(ANNSetup.Architecture == 'LSTM'):
        return LSTMNN(ANNSetup, test, train, DataSet.LVariables)
    elif(ANNSetup.Architecture == 'GRU'):
        return GRUNN(ANNSetup, test, train, DataSet.LVariables)


def LSTMNN(ANNSetup, test, train, VarList):
    """ Bulding a Keras for the Recurrent Neural Networks with LSTM layers """

    TrainWeights = GetTrainWeights(train.OutTrue,train.Weights) # Transformation of the Monte Carlo weights for training

    #Create the model and pass it the data for Callbacks
    model = Sequential()
    model.X_train = train.Events
    model.Y_train = train.OutTrue
    model.W_train = train.Weights       #Original weights!
    model.X_test  = test.Events
    model.Y_test  = test.OutTrue
    model.W_test  = test.Weights

    # Building the model from the predefinite configuration (RNN.py)
    LSTMNeurons  = ANNSetup.Neurons[0]
    DenseNeurons = ANNSetup.Neurons[1]
    width = train.Events.shape[1]
    Seq = train.Events.shape[2]
    model.add(LSTM(LSTMNeurons[0], input_shape=(width, Seq),kernel_regularizer=l1(ANNSetup.Regu), return_sequences=True))  # kernel_regularizer=l2(ANNSetup.Regu) 
    if(ANNSetup.Dropout[0] != 0):
        model.add(Dropout(ANNSetup.Dropout[0]))
    for i in range(1,len(LSTMNeurons)):
        if(i == len(LSTMNeurons)-1):                                               # Add last LSTMLayer
            model.add(LSTM(LSTMNeurons[i]))
        else:
            model.add(LSTM(LSTMNeurons[i],return_sequences=True,dropout=ANNSetup.Dropout[i], recurrent_regularizer=l2(ANNSetup.Regu)))
        if(ANNSetup.Dropout[i] != 0):
            model.add(Dropout(ANNSetup.Dropout[i]))
    for j in range(len(DenseNeurons)):
        model.add(Dense(DenseNeurons[j], activation='selu'))

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)    # Set the optimizer
    lrate = GetLearnRate(ANNSetup.LearnRate,ANNSetup.Epochs)    # Set a learning rate schedule
    Roc = Histories()                                           # Definite history for AUC at each training step
    if(lrate == None):
        Lcallbacks = [Roc]
    else:
        Lcallbacks = [Roc,lrate]


    model.summary()
    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])
    start = time.clock()                                        # start clock to track training time
    history = model.fit(train.Events, train.OutTrue, sample_weight=TrainWeights, validation_data=(test.Events, test.OutTrue, test.Weights), nb_epoch=int(ANNSetup.Epochs),
                        batch_size=int(ANNSetup.Batch), verbose=2, callbacks=Lcallbacks)
    end = time.clock()
    print("The training took {} seconds".format(end-start))

    LAuc = Roc.TestAucs
    LTrainAuc = Roc.TrainAucs
    print("Best Roc {0:.4f} at Epoch {1}".format(max(LAuc),LAuc.index(max(LAuc))+1)) #0:.4f
    print("Train Auc {0:.4f}".format(LTrainAuc[LAuc.index(max(LAuc))]))
    #print("Test Rocs: {0}".format(LAuc))

    for i in range(len(LAuc)):
        print("Auc at Epoch {0}: {1:.4f} Ov: {2:.3f}".format(i,LAuc[i],1-LAuc[i]/LTrainAuc[i]))

    model.save(ANNSetup.SavePath)

    return model, Roc


def GRUNN(ANNSetup, test, train, VarList):              #TODO: revise this!
    assert 0 == 1
    model = Sequential()
    GRUNeurons  = ANNSetup.Neurons[0]
    DenseNeurons = ANNSetup.Neurons[1]
    for i in range(len(GRUNeurons)):
        print(GRUNeurons[i])
        if(i == len(GRUNeurons)-1):
            model.add(GRU(GRUNeurons[i],activation='tanh', recurrent_activation='sigmoid'))     #,dropout=ANNSetup.Dropout[i]
        else:
            model.add(GRU(GRUNeurons[i],activation='tanh', recurrent_activation='sigmoid', return_sequences=True))                                                 
    for j in range(len(DenseNeurons)):
        model.add(Dense(DenseNeurons[j], activation='relu'))
    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate)


    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train.Events,train.OutTrue, sample_weight=train.Weights, nb_epoch=int(ANNSetup.Epochs), batch_size=int(ANNSetup.Batch), verbose=2)


    # model.save(ANNSetup.SavePath)
    # model.summary()

    return model


def GetOpti(Optimizer,LearnRate):
    """ Set optimizer as definite in the config """

    if(Optimizer == 'SGD'):
        Opti = optimizers.SGD(lr=LearnRate, momentum=0.0, nesterov=False)
    elif(Optimizer == 'Rmsprop'):
        Opti = optimizers.RMSprop(lr=LearnRate, rho=0.9)
    elif(Optimizer == 'Adagrad'):
        Opti = optimizers.Adagrad(lr=LearnRate)
    elif(Optimizer == 'Adam'):
        Opti = optimizers.Adam(lr=LearnRate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    return Opti

def GetLearnRate(DILr,Epochs):
    """ Using different Callbacks (definite in Callbacks.py) the learning rate schedule is set"""

    if(DILr.mode == 'poly'):
        ScheduelLr = PolynomialDecay(maxEpochs=DILr.StepSize,initAlpha=DILr.Lr,power=DILr.factor)
        ScheduelLr.plot(range(1,int(Epochs)+1))
        lrate = LearningRateScheduler(ScheduelLr)
    elif(DILr.mode == 'cycle'):
        lrate = CyclicLR(step_size=DILr.StepSize,mode=DILr.cycle,gamma=DILr.factor,base_lr=DILr.MinLr,max_lr=DILr.Lr)
    elif(DILr.mode == 'drop'):
        ScheduelLr = StepDecay(initAlpha=DILr.Lr, factor=DILr.factor, dropEvery=DILr.StepSize)
        ScheduelLr.plot(range(1,int(Epochs)+1))
        lrate = LearningRateScheduler(ScheduelLr)
    elif(DILr.mode == 'normal'):
        lrate = None

    return lrate

def GetTrainWeights(Labels,Weights):
    """ In some cases event weights are given by Monte Carlo generators, and may turn out to be overallvery small or large number.  To avoid artifacts due to this use renormalised weights.
        The  event  weights  are  renormalised such  that  both,  the  sum  of  all  weighted  signal  training  events  equals  the  sum  of  all  weights  ofthe background training events
        [https://arxiv.org/pdf/physics/0703039.pdf] """
    Weights = np.where(Weights > 0, Weights, 0)                 #Setting negative weights to zero for training
    ReferenceLength = len(Labels[Labels == 0])
    for Class in np.unique(Labels):
        CWeight = np.sum(Weights[Labels == Class])
        RenormFactor = ReferenceLength/CWeight
        Weights = np.where(Labels != Class,Weights,Weights*RenormFactor)

    return Weights











