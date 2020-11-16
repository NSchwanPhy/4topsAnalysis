import sys
sys.path.insert(1,'./srcFNN')
sys.path.insert(1,'./srcGeneral')

from tensorflow.python.client import device_lib
import tensorflow as tf
import SampleHandler
import FeedforwardNeuralNet as FNN
import EvalFNN
import DIClasses
import numpy as np
import os

""" Attention! The Signal Sample has to be first in the List of Samples i.e. the first in the Input Array to the NN else the inverse of the AUC is calculated """
np.random.seed(15)
#Parameters
Samples    = 'nomLoose'                                              # 'nomLoose' All samples, else List of Samples needed
ModelName  = 'FNN19'
BDT        = True                   # BDT model Name if None no Bdt should be trained
EvalFlag   = False                   # Should the models be evaluated? 
Odd        = False                  # Should the Odd combination be trained?
PreTrained = False                   # Was the Model allready evaluated befor?
SavePath   = './ANNOutput/FNN/' 

Type       = 'TMVA'                #FNN, TMVA, FNNMulti
Opti       = 'Adam'
Winit      = 'GlorotNormal'        #Lecun, Glorot, He
activation = 'elu'        #if number => alpha leaky relu
if(activation.replace('.','',1).isdigit()):
    activation = float(activation)
Epochs     = 1
Batch      = 10000
#Neurons  = [int(num) for num in sys.argv[1].split(',')]
Neurons    = [1,1]                 #Last Neuron should be 2 for TMVA and 1 for keras only implementation
Dropout    = None                               #No dropout -> None
Regu       = 0
Bootstrap  = ('test',None)                             #(Sample,Random seed for Bootstrap)
#LearnRate  = DIClasses.DILrSchedule('poly',0.004,factor=2,StepSize=Epochs)
#LearnRate = DIClasses.DILrSchedule('drop',0.005,factor=0.25,StepSize=10.)
#LearnRate = DIClasses.DILrSchedule('cycle',0.005,cycle='triangular',MinLr=0.001,StepSize=400.,factor=0.9)
LearnRate = DIClasses.DILrSchedule('normal',0.001)
# modes = {cycle,poly,drop,normal}, cycle = {triangular, triangular2, exp_range(factor=gamma)}

print("Neurons: {}".format(Neurons))
print("Epochs: {0}".format(Epochs))
print("Batch: {0}".format(Batch))
print("Dropout: {0}".format(Dropout))
print("Regu: {0}".format(Regu))
LearnRate.Print()


ListSamples = DIClasses.Init(ModelName,Samples,Cuts=True)
# Data Preparation
GPU  = False
Mode = 'Slow'                                               # Fast, Slow or Save
if(Samples != 'nomLoose'):
    Mode = 'Slow'
Sampler = SampleHandler.SampleHandler(ListSamples,mode=Mode+ModelName)
Sampler.norm    = False                                     # y axis to one                           
Sampler.valSize = 0.2                                       # Size of the validation sample
Sampler.Split   = 'EO'                                      # Use EO (even odd splitting)
Sampler.Plots   = False                                     # either False, LO or NLO            #TODO:Fix trafo problem
if(Type == 'TMVA'):
    Sampler.Scale    = None                                 
elif(Type == 'FNN' or 'FNNMulti'):
    Sampler.Scale    = 'ZScore'   
if(Sampler.Plots != False):
    Sampler.Scale   = None
if(GPU != True):
    tf.config.optimizer.set_jit(True)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
DeviceTyp = device_lib.list_local_devices()
DeviceTyp = str(DeviceTyp[0])
DeviceTyp = DeviceTyp[DeviceTyp.find("device_type:")+14:DeviceTyp.find("device_type:")+17]
print("Running on {0} device!".format(DeviceTyp))






DataSet = Sampler.GetANNInput()
#BDT Network Hyperparameters
BDTSetupEven = DIClasses.DIBDTSetup('BDTEven',800,3,5,2,30,8)
BDTSetupOdd  = DIClasses.DIBDTSetup('BDTOdd',800,3,5,2,30,8)

# Hand optiomalization
ModelNames = []
#Neural Network Hyperparameters
ANNSetupEven = DIClasses.DIANNSetup(Type,Epochs,SavePath+ModelName+'Even.h5',Batch,ModelName+'Even',LearnRate=LearnRate,Neurons=Neurons,Dropout=Dropout,Regu=Regu,Optimizer=Opti,
        Winit=Winit,Activ=activation)
ANNSetupOdd  = DIClasses.DIANNSetup(Type,Epochs,SavePath+ModelName+'Odd.h5',Batch,ModelName+'Odd',LearnRate=LearnRate,Neurons=Neurons,Dropout=Dropout,Regu=Regu,Optimizer=Opti,
        Winit=Winit,Activ=activation)
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
    

    



