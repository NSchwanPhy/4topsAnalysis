import numpy as np
import PlotService
from Guard import GuardFNN
from Callbacks import *
from Utils import *
from root_numpy.tmva import add_classification_events
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import optimizers
import time
import ROOT

import tensorflow as tf
#print(tf.__version__)

""" Thread control for usage on a cluster """
NUM_THREADS=1
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K

def Main(ANNSetup,DataSet,BDTSetup=None,BootStrap=('vali',None)):
    """
    Transforms the DataSet to a keras readable version and passes the configuration to the selected API

    ANNSetup:      Definition of the hyperparameters of the multivariant method
    DataSet:       Dataset created by the Sampler
    BDTSetup:      None if no BDT should be trained else True
    BootStrap:     Bootstrap setup tuble (Sample,random seed)

    Output:        trained model and Aucs for each training step
    """
    # Random seed for the NN cacluations
    np.random.seed(5)              
    tf.compat.v1.set_random_seed(5)

    train, test, vali = GetSamples(BootStrap,DataSet,ANNSetup.ModelName)    # Transform the DataSet into keras readable and aplying bootstrap if needed
    ANNSetup.InputDim = len(DataSet.LVariables)                             # Set Input dim (#features)
    GuardFNN(ANNSetup)                                                      # Protect the setup from typos

    # pass the Data to the correct API
    if(ANNSetup.Architecture == 'TMVA'):
        train, test, vali = GetSamples(BootStrap,DataSet,ANNSetup.ModelName,DoTrafo=False)    #In Utils
        dataloader, factory, output = Init(train, test, DataSet.LVariables)
        TMVAFNN(ANNSetup, dataloader, factory)
        if(BDTSetup != None):
            BDT(BDTSetup, dataloader, factory)
            #GetRocs(factory, dataloader,"BDTEven")
        Finialize(factory, output)
        #GetRocs(factory, dataloader,"FNN19Even")

    # Direct keras implementation
    elif(ANNSetup.Architecture == 'FNN'):
        train, test, vali = GetSamples(BootStrap,DataSet,ANNSetup.ModelName,DoTrafo=True)
        ANNSetup.InputDim = len(DataSet.LVariables)
        if(BDTSetup != None):
            stdwar("BDT is only supported using TMVA")
        return FNN(ANNSetup, test, train)

    # multi classifier (direct keras)
    elif(ANNSetup.Architecture == 'FNNMulti'):
        train, test, vali = GetSamples(BootStrap,DataSet,ANNSetup.ModelName,DoTrafo=True)
        ANNSetup.InputDim = len(DataSet.LVariables)
        if(BDTSetup != None):
            stdwar("BDT is only supported using TMVA")
        Model, Roc = MultiFNN(ANNSetup, test, train)
        if(BootStrap[1] != None):
            stdwar("BootStrap for multi has to be implemented")
            assert 0 == 1
        return Model, Roc


""" ------------------------------------------------------TMVA-----------------------------------------------------------------------------------"""
def Init(train,test,VarList):
    """ Implementation for TMVA Neural Network training """
    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    output = ROOT.TFile.Open('~/Data/NNOutput.root', 'RECREATE')
    factory = ROOT.TMVA.Factory('TMVAClassification', output,'!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
    dataloader = ROOT.TMVA.DataLoader('dataset')

    for Var in VarList:
        dataloader.AddVariable(Var)

    add_classification_events(dataloader,train.Events,train.OutTrue,weights=train.Weights,signal_label=1)
    add_classification_events(dataloader,test.Events,test.OutTrue,weights=test.Weights,signal_label=1,test=True)

    dataloader.PrepareTrainingAndTestTree(ROOT.TCut(''),'SplitSeed=100')    #:NormMode=None
    #CrossCheck(dataloader)

    return dataloader, factory , output

def TMVAFNN(ANNSetup, dataloader, factory):
    """ Build the keras model used in TMVA """

    model = Sequential()
    model.add(Dense(ANNSetup.Neurons[0], activation='selu', input_dim=ANNSetup.InputDim))
    if(ANNSetup.Dropout != None):
        model.add(Dropout(ANNSetup.Dropout))
    for i in range(1,len(ANNSetup.Neurons)):
        if(i == len(ANNSetup.Neurons)-1):
            model.add(Dense(ANNSetup.Neurons[i], activation='sigmoid'))
        else:
            model.add(Dense(ANNSetup.Neurons[i], activation='selu'))

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)
    #Compiling the model
    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])

    model.save(ANNSetup.SavePath)
    #model.summary()
    
    factory.BookMethod(dataloader,ROOT.TMVA.Types.kPyKeras, ANNSetup.ModelName,
    '!H:!V:FilenameModel='+ANNSetup.SavePath+':NumEpochs='+ANNSetup.Epochs+':BatchSize='+ANNSetup.Batch+":VarTransform=G")        #:VarTransform=G

    return

def BDT(BDTSetup, dataloader, factory):
    """ BDT defintion in TMVA """

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT,
                        BDTSetup.ModelName ,"!H:!V:NTrees="+BDTSetup.TreeNumber+":MinNodeSize="+BDTSetup.NMinActual+"%:BoostType=Grad:Shrinkage="+BDTSetup.Shrinkage+
                        ":UseBaggedBoost:BaggedSampleFraction="+BDTSetup.BaggingActual+":nCuts="+BDTSetup.NCutActual+":MaxDepth="+BDTSetup.MaxDepth+
                        ":IgnoreNegWeightsInTraining=True" )

    return

def Finialize(factory, output):
    """ training, testing and Evaluation off all models definited in TMVA """

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    output.Close()

    return

def GetRocs(factory, dataloader,name):
    """ Calcuation of the AUC with TMVA """
    print(name)
    f = ROOT.TFile('/gpfs/share/home/s6nsschw/Data/NNOutput.root')
    TH_Train_S = f.Get("/dataset/Method_"+name+"/"+name+"/MVA_"+name+"_Train_S")
    TH_Train_B = f.Get("/dataset/Method_"+name+"/"+name+"/MVA_"+name+"_Train_B")

    RocSame = ROOT.TMVA.ROCCalc(TH_Train_S,TH_Train_B)
    AucSame = RocSame.GetROCIntegral()
    print("Auc test")
    print(factory.GetROCIntegral(dataloader,name))       #Auc test
    print("Auc train")
    print(AucSame)                                              #Auc train

    del RocSame
    f.Close()

    return



def CrossCheck(dataloader):

    DataSet = dataloader.GetDataSetInfo().GetDataSet()                
    EventCollection = DataSet.GetEventCollection()
    BkgW, SigW = np.zeros([]), np.zeros([])
    Bkg, Sig = np.zeros([]), np.zeros([])
    for Event in EventCollection:
        if(Event.GetClass() == 1):
            Bkg  = np.append(Bkg, Event.GetValue(1)) 
            BkgW = np.append(BkgW, Event.GetWeight())
        elif(Event.GetClass() == 0):
            Sig = np.append(Sig, Event.GetValue(1))
            SigW = np.append(SigW, Event.GetWeight())
    PlotService.VarCrossCheck(Sig,Bkg,SigW,BkgW,'njets',-6,4,10)

    return


""" ---------------------------------------------- KERAS ---------------------------------------------------------------- """

def FNN(ANNSetup, test, train):
    """ Bulding a Keras for the Feedforward Neural Networks """

    #ClassWeights = GetClassWeights(train.OutTrue,train.Weights)
    TrainWeights = GetTrainWeights(train.OutTrue,train.Weights)    # Transformation of the Monte Carlo weights for training

    #tf.debugging.set_log_device_placement(True)                   #Check if system is running on the correct device

    #Create the model and pass it the data for Callbacks
    model = Sequential()
    model.X_train = train.Events
    model.Y_train = train.OutTrue
    model.W_train = train.Weights       #Original weights!
    model.X_test  = test.Events
    model.Y_test  = test.OutTrue
    model.W_test  = test.Weights

    model = BuildModel(ANNSetup,model)                          # Building the model from the predefinite configuration (FNN.py)

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)    # Set the optimizer
    lrate = GetLearnRate(ANNSetup.LearnRate,ANNSetup.Epochs)    # Set a learning rate schedule
    Roc = Histories()                                           # Definie history for the AUC results at each training step 
    #Roc = RedHistory()
    if(lrate == None):
        Lcallbacks = [Roc]
        #Lcallbacks = []
    else:
        Lcallbacks = [Roc,lrate]
        #Lcallbacks = [lrate]

    model.summary()
    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])
    start = time.clock()                                        # start clock to track training time
    history = model.fit(train.Events,train.OutTrue, sample_weight=TrainWeights, validation_data=(test.Events, test.OutTrue, test.Weights), epochs=int(ANNSetup.Epochs),
                        batch_size=int(ANNSetup.Batch), verbose=2, callbacks=Lcallbacks)                  #, callbacks=Lcallbacks , sample_weight=TrainWeights
    #history = model.fit(train.Events,train.OutTrue,batch_size=4000,epochs=2)
    end = time.clock()
    print("The training took {} seconds".format(end-start))

    LAuc = Roc.TestAucs
    LTrainAuc = Roc.TrainAucs
    print("Best Test Auc {0:.4f} at Epoch {1}".format(max(LAuc),(LAuc.index(max(LAuc))+1))) #0:.4f
    print("Best Train Auc {0:.4f}".format(LTrainAuc[LAuc.index(max(LAuc))]))

    for i in range(len(LAuc)):
        print("Auc at Epoch {0}: {1:.4f} Ov: {2:.3f}".format(i,LAuc[i],1-LAuc[i]/LTrainAuc[i]))

    model.save(ANNSetup.SavePath)

    return model, Roc


def FastAUC(model):
    """ Calculate AUC (debug method) """

    train_pred = model.predict(model.X_train)
    test_pred  = model.predict(model.X_test)
    return roc_auc_score(model.Y_train,train_pred), roc_auc_score(model.Y_test, test_pred)


def MultiFNN(ANNSetup, test, train):
    """ Bulding a Keras for multi-classifer """

    #One hot encoding
    TrainMultiClass   = to_categorical(train.MultiClass)
    TestMultiClass   = to_categorical(test.MultiClass)

    #ClassWeights = GetClassWeights(train.MultiClass,train.Weights)
    TrainWeights = GetTrainWeights(train.MultiClass,train.Weights)              # Transformation of the Monte Carlo weights for training

    #Create the model and pass it the data for Callbacks
    model = Sequential()
    model.Y_test  = TestMultiClass[:,0]
    model.X_train = train.Events
    model.Y_train = TrainMultiClass[:,0]
    model.W_train = train.Weights       #Original weights!

    # Build model from configuration (set in FNN.py)
    model.add(Dense(ANNSetup.Neurons[0], activation='selu', input_dim=ANNSetup.InputDim))
    if(ANNSetup.Dropout != None):
        model.add(Dropout(ANNSetup.Dropout))
    for i in range(1,len(ANNSetup.Neurons)):
        if(i == len(ANNSetup.Neurons)-1):
            model.add(Dense(ANNSetup.Neurons[i], activation='softmax'))
        else:
            model.add(Dense(ANNSetup.Neurons[i], activation='selu'))

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)        # Set optimizer
    lrate = GetLearnRate(ANNSetup.LearnRate,ANNSetup.Epochs)        # Set learning rate schedule
    Roc = Histories()                                               # Create history for AUC during training
    Lcallbacks = [Roc,lrate]

    model.compile(optimizer=Opti, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train.Events, TrainMultiClass, sample_weight=TrainWeights, validation_data=(test.Events, TestMultiClass, test.Weights), epochs=int(ANNSetup.Epochs),
                        batch_size=int(ANNSetup.Batch), verbose=2, callbacks=Lcallbacks)            #, sample_weight=TrainWeights

    LAuc = Roc.TestAucs
    LTrainAuc = Roc.TrainAucs
    print("Best Roc {0:.4f} at Epoch {1}".format(max(LAuc),LAuc.index(max(LAuc))+1))
    print("Train Auc {0:.4f}".format(LTrainAuc[LAuc.index(max(LAuc))]))
    # print("Test Rocs: {0}".format(LAuc))
    # print("Test Loss: {0}".format(Roc.TestLosses))
    # print("Train Rocs: {0}".format(LTrainAuc))
    # print("Train Loss: {0}".format(Roc.TrainLosses))

    model.save(ANNSetup.SavePath)

    return model, Roc



def CrossCheck(dataloader):
    """ make a njets plots to check data gets imported in the right way """




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

    return np.asarray(lrate)

def GetClassWeights(Labels,Weights):
    """ Calculation of class weights (weight all samples to the same Yield)"""
    ClassWeight = {}
    for Class in np.unique(Labels):
        TotalWeight = np.sum(Weights[Labels == Class])
        CWeight = np.sum(Weights)/(len(np.unique(Labels))*TotalWeight)
        ClassWeight.update({int(Class):CWeight})

    print(ClassWeight)
    return ClassWeight

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


def Winit(Weightinit):
    """ Look up for weight initialization """

    DicWinit = {'LecunNormal':tf.keras.initializers.lecun_normal(seed=None),
            'LecunUniform':tf.keras.initializers.lecun_uniform(seed=None),
            'GlorotNormal':tf.keras.initializers.GlorotNormal(seed=None),
            'GlorotUniform':tf.keras.initializers.GlorotUniform(seed=None),
            'HeNormal':tf.keras.initializers.he_normal(seed=None),
            'HeUniform':tf.keras.initializers.he_uniform(seed=None)}
    return DicWinit[Weightinit]


def BuildModel(ANNSetup,model):
    """  Building model from config (definite in FNN.py) """

    if(isinstance(ANNSetup.Activ,str)):
        model.add(Dense(ANNSetup.Neurons[0], kernel_regularizer=l2(ANNSetup.Regu), activation=ANNSetup.Activ, kernel_initializer=Winit(ANNSetup.Winit), input_dim=ANNSetup.InputDim))
        if(ANNSetup.Dropout != None):
            model.add(Dropout(ANNSetup.Dropout))
        for i in range(1,len(ANNSetup.Neurons)):
            if(i == len(ANNSetup.Neurons)-1):
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit), activation='sigmoid'))
            else:
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit), activation=ANNSetup.Activ))
    else:
        model.add(Dense(ANNSetup.Neurons[0], kernel_regularizer=l2(ANNSetup.Regu), kernel_initializer=Winit(ANNSetup.Winit), input_dim=ANNSetup.InputDim))
        model.add(LeakyReLU(alpha=ANNSetup.Activ))
        if(ANNSetup.Dropout != None):
            model.add(Dropout(ANNSetup.Dropout))
        for i in range(1,len(ANNSetup.Neurons)):
            if(i == len(ANNSetup.Neurons)-1):
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit), activation='sigmoid'))
            else:
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit)))
                model.add(LeakyReLU(alpha=ANNSetup.Activ))

    return model



def FastAUC(model):
    
    train_pred = model.predict(model.X_train)
    test_pred  = model.predict(model.X_test)
    return roc_auc_score(model.Y_train,train_pred), roc_auc_score(model.Y_test, test_pred)






