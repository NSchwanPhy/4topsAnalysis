import sys
sys.path.insert(1,'./srcFNN')
sys.path.insert(1,'./srcGeneral')

import tensorflow as tf
import SampleHandler
import FeedforwardNeuralNet as FNN
import EvalFNN
import DIClasses
import Utils
import numpy as np
import os

""" Analysis Setup """
np.random.seed(15)                  #Random seed used for splitting
Samples    = 'nomLoose'             # 'nomLoose' All samples, else List of Samples needed
ModelName  = 'FNN18'                # Set of Variables (definite in DIClasses Init function)
BDT        = None                   # BDT model Name if None no Bdt is trained (TMVA only)
EvalFlag   = False                  # Evaluation of the model
Odd        = False                  # Training of the Odd model (exchange of the training and testing set)
PreTrained = False                  # Evaluate a model previously trained 
SavePath   = './ANNOutput/FNN/'     # NN trainings data output save path

""" Neural Network Parameters """
Type       = 'FNN'                 #FNN (Keras), TMVA, FNNMulti
Opti       = 'Adam'                #Adam, Rmsprop, SGD, Adagrad
Winit      = 'GlorotNormal'        #Lecun, Glorot, He (Normal or Uniform)
activation = 'elu'                 #if number => alpha leaky relu elif selu, relu, elu etc.
if(activation.replace('.','',1).isdigit()):
    activation = float(activation)
Epochs     = 1
Batch      = 10000
#Neurons  = [int(num) for num in sys.argv[1].split(',')]
Neurons    = [1,1]                 #Last layer should contain 2 Neurons for TMVA, 1 for FNN , or #classes for FNNMulti
Dropout    = None                  #prop. of Dropout (No dropout -> None)
Regu       = 0                     #l2 regularization
Bootstrap  = ('test',None)         #Bootstrap option for error evaluation (Sample,Random seed for Bootstrap)

"""" Learning rate scheduel """
#LearnRate  = DIClasses.DILrSchedule('poly',0.004,factor=2,StepSize=Epochs)
#LearnRate = DIClasses.DILrSchedule('drop',0.005,factor=0.25,StepSize=10.)
#LearnRate = DIClasses.DILrSchedule('cycle',0.005,cycle='triangular',MinLr=0.001,StepSize=400.,factor=0.9)
LearnRate = DIClasses.DILrSchedule('normal',0.001)
# modes = {cycle,poly,drop,normal}, cycle = {triangular, triangular2, exp_range(factor=gamma)}

Utils.stdinfo("Neurons: {}".format(Neurons))
Utils.stdinfo("Epochs: {0}".format(Epochs))
Utils.stdinfo("Batch: {0}".format(Batch))
Utils.stdinfo("Dropout: {0}".format(Dropout))
Utils.stdinfo("Regu: {0}".format(Regu))
LearnRate.Print()

ListSamples = DIClasses.Init(ModelName,Samples,Cuts=True)   # Initiate the setup to import the samples (Cuts string of singal region cuts or True for standard definition)

GPU  = False                                                # Enable GPU training
Mode = 'Fast'                                               # Fast, Slow or Save
if(Samples != 'nomLoose'):
    Mode = 'Slow'
Sampler = SampleHandler.SampleHandler(ListSamples,mode=Mode+ModelName)  # Initiate the Sampler
Sampler.NormFlag    = False                                 # y axis normalized to 1                           
Sampler.valSize = 0.2                                       # Size of the validation sample
Sampler.Split   = 'EO'                                      # Use EO (even odd splitting)
Sampler.Plots   = False                                     # Which sample should be used for the signal (LO or NLO), no plots => False
if(Type == 'TMVA'):
    Sampler.TrafoFlag    = None                                 
elif(Type == 'FNN' or 'FNNMulti'):
    Sampler.TrafoFlag    = 'ZScore'                         # Transformation of the input features (ZScore => normally distributed)
if(Sampler.Plots != False):
    Sampler.TrafoFlag   = None
if(GPU != True):
    tf.config.optimizer.set_jit(True)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
DeviceTyp = tf.config.experimental.list_physical_devices()
if('GPU' in str(DeviceTyp)):
    Utils.stdwar("GPU training enabled!")
else:
    Utils.stdwar("CPU training!")








DataSet = Sampler.GetANNInput()                                    # Process the samples using the setup definite previously (ListSamples)
""" BDT and NN hyperparameters """
#BDT hyperparameters
BDTSetupEven = DIClasses.DIBDTSetup('BDTEven',800,3,5,2,30,8)      
BDTSetupOdd  = DIClasses.DIBDTSetup('BDTOdd',800,3,5,2,30,8)
ModelNames = []
#Neural Network hyperparameters
ANNSetupEven = DIClasses.DIANNSetup(Type,Epochs,SavePath+ModelName+'Even.h5',Batch,ModelName+'Even',LearnRate=LearnRate,Neurons=Neurons,Dropout=Dropout,Regu=Regu,Optimizer=Opti,
        Winit=Winit,Activ=activation)
ANNSetupOdd  = DIClasses.DIANNSetup(Type,Epochs,SavePath+ModelName+'Odd.h5',Batch,ModelName+'Odd',LearnRate=LearnRate,Neurons=Neurons,Dropout=Dropout,Regu=Regu,Optimizer=Opti,
        Winit=Winit,Activ=activation)

""" Implemetation of the flags set above"""
if(PreTrained == False):
    if(BDT != None):
        ModelNames.append(BDTSetupEven.ModelName)
        ModelNames.append(ANNSetupEven.ModelName)
        ModelEven, RocEven = FNN.Main(ANNSetupEven, DataSet, BDTSetup=BDTSetupEven, BootStrap=Bootstrap)
        ModelOdd, RocOdd = None, None
        if(Odd == True):
            ModelNames.append(BDTSetupOdd.ModelName)
            ModelNames.append(ANNSetupOdd.ModelName)
            ModelOdd, RocOdd = FNN.Main(ANNSetupOdd, DataSet, BDTSetup=BDTSetupOdd, BootStrap=Bootstrap)
    elif(BDT == None):
        ModelNames.append(ANNSetupEven.ModelName)
        ModelEven, RocEven = FNN.Main(ANNSetupEven, DataSet, BootStrap=Bootstrap)
        ModelOdd, RocOdd = None, None
        if(Odd == True):
            ModelNames.append(ANNSetupOdd.ModelName)
            ModelOdd, RocOdd = FNN.Main(ANNSetupOdd, DataSet, BootStrap=Bootstrap)
    if(EvalFlag == True):
        Eval = EvalFNN.EvalFNN(Type,SavePath,ModelNames,DataSet,ModelEven,ModelOdd,RocEven,RocOdd)
        Eval.EvaluateFNN()

    if(Type == 'TMVA'):
        os.system("mv ./dataset/weights/*.xml ./ANNOutput/FNN/")
        os.system("mv ./dataset/weights/*.h5 ./ANNOutput/FNN/")

else:
    if(BDT != None):
        ModelNames.append(BDTSetupEven.ModelName)
        ModelNames.append(ANNSetupEven.ModelName)
        if(Odd == True):
            ModelNames.append(BDTSetupOdd.ModelName)
            ModelNames.append(ANNSetupOdd.ModelName)
    elif(BDT == None):
        ModelNames.append(ANNSetupEven.ModelName)
        if(Odd == True):
            ModelNames.append(ANNSetupOdd.ModelName)

    if(EvalFlag == True):
        Eval = EvalFNN.EvalFNN(Type,SavePath,ModelNames,DataSet,None,None,None,None)
        Eval.PreTrainedFNN()  
    

    



