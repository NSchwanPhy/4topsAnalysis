import sys
sys.path.insert(1,'./srcRNN')
sys.path.insert(1,'./srcGeneral')

from tensorflow.python.client import device_lib
import tensorflow as tf
import Utils
import SampleHandler
import RecurrentNeuralNet
import EvalRNN
import numpy as np
import os
import DIClasses
import tensorflow as tf
import Utils

""" Analysis Setup """
np.random.seed(15)                  # Random seed used for splitting
Samples    = 'nomLoose'             # 'nomLoose' All samples, else List of Samples needed
ModelName   = 'LSTM'                # Set of Variables (definite in DIClasses Init function)
EvalFlag   = False                  # Evaluation of the model
Odd        = False                  # Training of the Odd model (exchange of the training and testing set)
PreTrained = False                  # Evaluate a model previously trained 
SavePath    = './ANNOutput/RNN/'    # NN trainings data output save path

""" Neural Network Parameters """
LayerType = 'LSTM'                  # LSTM or GRU
Optimizer = 'Adam'                  #Adam, Rmsprop, SGD, Adagrad
Epochs  = 10
Batch   = 500
Neurons = [[64,1],[]]                 # Last Layer needs to be have 1 Neuron
Dropout = [0]                         # List of dropout prop. per layer
Regu = 0.001                          # l1 regularization term
Bootstrap = ('test',None)             # Bootstrap option for error evaluation (Sample,Random seed for Bootstrap)
Dropout.extend([0] * (len(Neurons[0])-len(Dropout)))

"""" Learning rate scheduel """
LearnRate  = DIClasses.DILrSchedule('poly',0.004,factor=2,StepSize=Epochs)
#LearnRate = DIClasses.DILrSchedule('normal',0.001)

print("LayerType: {}".format(LayerType))
print("Neurons: {}".format(Neurons))
print("Epochs: {0}".format(Epochs))
print("Batch: {0}".format(Batch))
print("Dropout: {0}".format(Dropout))
print("Regu: {0}".format(Regu))
LearnRate.Print()

ListSamples = DIClasses.Init(ModelName,Samples,Cuts=True) # Initiate the setup to import the samples (Cuts string of singal region cuts or True for standard definition)

GPU  = True                                               # Enable for GPU training
Mode = 'Fast'                                             # Fast, Slow or Save
if(Samples != 'nomLoose'):
    Mode = 'Save'
Sampler = SampleHandler.SampleHandler(ListSamples,mode=Mode+ModelName) # Initiate the Sampler
Sampler.norm    = False                                   # Y axis norm to one 
Sampler.valSize = 0.2                                     # Size of the validation sample
Sampler.Split   = 'EO'                                    # Use EO (even odd splitting)
Sampler.Scale   = 'ZScoreLSTM'                            # Kind of Norm use ZScoreLTSM
Sampler.SequenceLength  = 7                               # Length of the (time) sequence of a bach (#jets per event)
Sampler.Plots = 'NLO'                                     # Which sample should be used for the signal (LO or NLO), no plots => False
if(GPU != True):
    tf.config.optimizer.set_jit(True)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
DeviceTyp = device_lib.list_local_devices()
DeviceTyp = str(DeviceTyp[0])
DeviceTyp = DeviceTyp[DeviceTyp.find("device_type:")+14:DeviceTyp.find("device_type:")+17]
Utils.stdwar("Running on {0} device!".format(DeviceTyp))





DataSet       = Sampler.GetANNInput()                   # Process the samples using the setup definite previously (ListSamples)
#Neural Network Hyperparameters
ANNSetupEven = DIClasses.DIANNSetup(LayerType,Epochs,SavePath+ModelName+'Even.h5',Batch,ModelName+'Even',Neurons,Dropout=Dropout,LearnRate=LearnRate,Optimizer=Optimizer,Regu=Regu)
ANNSetupOdd  = DIClasses.DIANNSetup(LayerType,Epochs,SavePath+ModelName+'Odd.h5',Batch,ModelName+'Odd',Neurons,Dropout=Dropout,LearnRate=LearnRate,Optimizer=Optimizer,Regu=Regu)
ModelNames = [ANNSetupEven.ModelName,'BDTEven5','BDTEven19']

""" Implemetation of the flags set above"""
if(PreTrained == False):
    #Even: Training and Evaluation
    ModelEven, RocEven = RecurrentNeuralNet.Main(ANNSetupEven, DataSet, BootStrap=Bootstrap)
    ModelOdd, RocOdd = None, None

    #Odd: Training and Evaluation
    if(Odd == True):
        ModelOdd, RocOdd  = RecurrentNeuralNet.Main(ANNSetupOdd, DataSet, BootStrap=Bootstrap)                              
        ModelNames.append(ANNSetupOdd.ModelName)


    if(EvalFlag == True):
        Eval = EvalRNN.EvalRNN(SavePath,ModelNames,DataSet,ModelEven,ModelOdd,RocEven,RocOdd)
        Eval.EvaluateRNN()

elif(PreTrained == True):
        Eval = EvalRNN.EvalRNN(SavePath,ModelNames,DataSet,None,None,None,None)
        Eval.PreTrainedRNN()
        













