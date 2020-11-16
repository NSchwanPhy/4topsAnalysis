import ROOT
import copy
import numpy as np
from math import sqrt
import PlotService
from DIClasses import DISample, DIDataSet
import Utils, os
from root_numpy import tree2array, root2array, array2root
from sklearn.model_selection import train_test_split


class SampleHandler:

    TrafoFlag       = None
    valSize         = 0.2
    Split           = 'EO'
    NormFlag        = True
    SequenceLength  = 1
    Plots           = True

    def __init__(self, ListAnaSetup, mode='Slow', Single=False):
        self.ListAnaSetup = ListAnaSetup
        self.mode         = mode
        self.Single       = Single
        self.Means        = []
        self.Stds         = []

    def GetANNInput(self,verbose=1):

        if(self.mode[:4] == 'Slow' or self.mode[:4] == 'Save'):
            self.verbose = verbose
            if(self.verbose == 1):
                Utils.stdinfo("Setting up input vectors")
            for DISetup in self.ListAnaSetup:
                if(self.verbose == 1):
                    print("Processing Sample: "+DISetup.Name)
                ListOfVariables = DISetup.LVars[:]
                if(self.mode[:4] == 'Fast' and DISetup.Name != 'NLO'):
                    ListOfVariables.append('Weight')
                else:
                    ListOfVariables.extend(DISetup.WeightList)
                Arr = self.GetArray(ListOfVariables,DISetup)
                np.random.shuffle(Arr)
                DISetup.Samples = self.MakeSplit(Arr,DISetup)
                if(self.verbose == 1):
                    self.Info(DISetup)
            if(self.verbose == 1):
                Utils.stdinfo("finalising input Preparation")
            ListSamples = [DISetup.Samples for DISetup in self.ListAnaSetup]
            train = self.Finalise("train",ListSamples)
            test  = self.Finalise("test",ListSamples)
            vali  = self.Finalise("validation",ListSamples)
            DataSet = DIDataSet(train,test,vali)

        if(self.mode[:4] == 'Save'):
            if(self.verbose == 1):
                Utils.stdinfo("Saving data for Fast mode")
            SaveNpy('TrainSet'+self.mode[4:],train)
            SaveNpy('TestSet'+self.mode[4:],test)
            SaveNpy('ValiSet'+self.mode[4:],vali)
            
        elif(self.mode[:4] == 'Fast'):
            train = self.GetSampleNpy('TrainSet'+self.mode[4:])
            test = self.GetSampleNpy('TestSet'+self.mode[4:])
            vali = self.GetSampleNpy('ValiSet'+self.mode[4:])
            DataSet = DIDataSet(train,test,vali)

        if(self.Plots != False):
            PlotService.VarHists(DataSet,key='All',Norm=self.NormFlag,Sig=self.Plots) 
            assert 0 == 1   


        return DataSet
    
    def Finalise(self, key,ListSamples):
        # concatenate all Bkg and Signal Samples for the input
        LSample = [Samples[key] for Samples in ListSamples]
        LSample = list(filter(None, LSample))
        listAllEvents     = [Sample.Events for Sample in LSample]
        listAllWeights    = [Sample.Weights for Sample in LSample]
        listAllOutTrue    = [Sample.OutTrue for Sample in LSample]
        listAllMultiClass = []
        Names = [DISetup.Name for DISetup in self.ListAnaSetup]                            #For Multiclass tttt is 0 (in Binary 1)                                   
        # for i,Sample in enumerate(listAllOutTrue):
        #     if(Names[i] in ['ttW','ttH','ttZ']):
        #         MultiClass = np.full((len(Sample)),1)
        #     elif(Names[i] in ["tttt"]):
        #         MultiClass = np.full((len(Sample)),0)
        #     else:
        #         MultiClass = np.full((len(Sample)),2)
        #     listAllMultiClass.append(MultiClass)

        ClassNum = 0         
        for i,Sample in enumerate(listAllOutTrue):                                                    #14 Classes
            MultiClass = np.full((len(Sample)),ClassNum)
            listAllMultiClass.append(MultiClass)                                                                               #NLO and LO are both singal
            ClassNum += 1


        AllWeights      = np.concatenate(listAllWeights,axis=0)
        AllOutTrue      = np.concatenate(listAllOutTrue,axis=0)
        AllMultiClass   = np.concatenate(listAllMultiClass,axis=0)
        AllEvents       = np.concatenate(listAllEvents,axis=0)

        if(key == "train"):                                                             #Checking for negative Weights in the Sample                                                            
            for i, Sample in enumerate(LSample):
                NWeights = 0
                for weight in Sample.Weights:
                    if(weight < 0):
                        NWeights += 1
                if(NWeights != 0 and self.verbose == 1):
                    Utils.stdwar("We have Negative weights! {0} in {1} (train)".format(NWeights, self.ListAnaSetup[i].Name))

        Sample = DISample(AllEvents,AllWeights,AllOutTrue,AllMultiClass,self.ListAnaSetup[0].LVars,Names)
        #self.ShuffleInUnison(Sample)

        return Sample


    def CleanUp(self,key):
        #TODO: use the DataSet function OneSample (does it work?). Transform usw up date dataset

        AllEvents = self.Trafo(AllEvents,key)                                            # Setting the x-axis scalling
        if(np.ndim(AllEvents) == 3):                                                    # events, t, Var
            AllEvents = np.swapaxes(AllEvents,1,2)
        AllEvents = np.where(AllEvents != -3.4*pow(10,38), AllEvents, 0)                # replacing all padded element with 0

            
    def GetArray(self,ListOfVariables,DISetup):
        
        for i, path in enumerate(DISetup.Path):
            if(DISetup.Cuts == ''):
                selection = DISetup.McChannel
            elif(DISetup.McChannel == ''):
                selection = DISetup.Cuts
            else:
                selection = "("+DISetup.McChannel+") && ("+DISetup.Cuts+")"

            # get the array from the tree
            rfile  = ROOT.TFile(path)
            intree = rfile.Get(DISetup.Tree)
            Arr = tree2array(intree,branches=ListOfVariables,selection=selection)
            Arr = np.array(Arr.tolist(), dtype=object)
            if(i == 0):
                TotalArr = Arr
            elif(Arr.shape != (0,)):
                TotalArr = np.append(TotalArr,Arr,axis=0)

        TotalArr = Utils.ConvertArr(TotalArr,self.SequenceLength)
        return np.array(TotalArr, dtype=np.float64)

    def GetSampleNpy(self,fname):
        fpath = os.path.expanduser("~")
        if(fpath == "/jwd"):
              fpath = "/cephfs/user/s6nsschw"
        fpath = fpath+"/Data/Fast/"

        Events = np.load(fpath+fname+"_Events.npy", allow_pickle=True)
        Other = np.load(fpath+fname+"_Other.npy", allow_pickle=True)
        Weights = Other[:,0]
        OutTrue = np.array(Other[:,1],dtype=int)
        MultiClass = np.array(Other[:,2],dtype=int)
        
        Names = [DISetup.Name for DISetup in self.ListAnaSetup]
        return DISample(Events,Weights,OutTrue,MultiClass,self.ListAnaSetup[0].LVars,Names)

    def Trafo(self,Events,Sampletype):
        if(self.TrafoFlag == 'MinMax'):
            Utils.stdwar("Min not implemented for LSTM")
            for i in range(len(Events[0])):
                min, max = self.GetMinMax(Events,i)                 #TODO: use training min max for test
                for j in range(len(Events)):
                    Events[j][i] = float(Events[j][i])
                    Events[j][i] = (Events[j][i] - min)/(max - min) - 0.5
        elif(self.TrafoFlag == 'ZScoreLSTM'):
            for iVar in range(Events.shape[1]):                                                  #Loop over Variable
                if(Sampletype == 'train'):                                                #Use same var and mean for all sets
                    Var = np.ma.masked_equal(Events[:,iVar],-3.4*pow(10,38))
                    self.Means.append(np.mean(Var.flatten()))
                    self.Stds.append(np.var(Var.flatten()))
                for iSeq in range(Events.shape[2]):                                           #Loop over Sequence
                        for Event in range(len(Events)):                                                #Loop over Batch
                            if(Events[Event][iVar][iSeq] != -3.4*pow(10,38)):
                                Events[Event][iVar][iSeq] = float(Events[Event][iVar][iSeq])
                                Events[Event][iVar][iSeq] = (Events[Event][iVar][iSeq] - self.Means[iVar])/sqrt(self.Stds[iVar])
        elif(self.TrafoFlag == 'ZScore'):
            for iVar in range(Events.shape[1]):                                                  #Loop over Variable
                if(Sampletype == 'train'):
                    mean, variance = np.mean(Events[:,iVar]), np.var(Events[:,iVar])
                    self.Means.append(mean)
                    self.Stds.append(variance)
                else:
                    mean = self.Means[iVar]
                    variance = self.Stds[iVar]
                if(variance == 0 and mean !=0):
                    Utils.stdwar("All Entries in this variable are the same and not equal 0. Skipping!")
                    Utils.stdwar("Variableidx {0}".format(iVar))
                elif(variance != 0):
                    for iBatch in range(len(Events)):                                                #Loop over Batch
                        Events[iBatch][iVar] = float(Events[iBatch][iVar])
                        Events[iBatch][iVar] = (Events[iBatch][iVar] - mean)/sqrt(variance)            
            
        elif(self.TrafoFlag != None):
            Utils.stdwar("This norm is not in implemented!")
            assert 0 == 1

        return Events

    def GetMinMax(self,Arr,col):
        return Arr.min(0)[col], Arr.max(0)[col]

    def MakeSplit(self, Arr, DISetup):
        if(DISetup.Name != "tttt"):                                                 #20% of all samples are used to validate but the LO Sample (tttt)
            validation, Arr      = self.MakeValidation(Arr, self.valSize)
        else:                                                                       #Assign Value for tttt
            validation = None
        if(DISetup.Name != "tttt" and DISetup.Name != "NLO"):                       #Split background samples into 40% test 40% train
            test, train          = self.MakeTestTrain(Arr, self.Split)
        elif(DISetup.Name == "tttt"):                                               #Only train on LO
            test  = None
            train = Arr
        elif(DISetup.Name == "NLO"):                                                #Only test on NLO
            test  = Arr
            train = None
        else:
            Utils.stderr("Bullshit sample!")
            assert 0 == 1
        Sample               = {"train": train, "test": test, "validation": validation}
        Sample["train"]      = self.MakeDISample(train, DISetup)
        Sample["test"]       = self.MakeDISample(test, DISetup)
        Sample["validation"] = self.MakeDISample(validation, DISetup)
        return Sample


    def MakeValidation(self, Arr, valSize=0.2):
        splitIndex = int(round(Arr.shape[0] * (1-valSize)))
        return Arr[splitIndex:], Arr[:splitIndex]
    

    def MakeTestTrain(self,Arr,Split='EO'):
        if(Split == 'EO'):
            train = Arr[::2]
            test  = Arr[1::2]
        elif(Split == 'H'):                                            
            train  = Arr[:len(Arr)/2]
            test = Arr[len(Arr)/2:]
        elif(Split == 'K'):
            train, test = train_test_split(Arr)
        return test, train

    def MakeDISample(self,Arr,DISetup):
        if(not isinstance(Arr,np.ndarray)):
            return None
        else:
            Events   = Arr[:,:len(DISetup.LVars)]
            # Events   = Utils.ConvertArr(Events,self.SequenceLength)
            Weights  = Arr[:,len(DISetup.LVars):]
            # Weights  = Utils.ConvertArr(Weights,1)
            Weights  = self.GetWeightSum(Weights,DISetup)                         
            if(DISetup.Name == "tttt" or DISetup.Name == "NLO"):
                OutTrue = np.ones([len(Events)])
            else:
                OutTrue = np.zeros([len(Events)])
            Names = [DISetup.Name for DISetup in self.ListAnaSetup]
            return DISample(Events,Weights,OutTrue,None,self.ListAnaSetup[0].LVars,Names)

    def Info(self, DISetup):
        train = DISetup.Samples["train"]
        test  = DISetup.Samples["test"]
        vali  = DISetup.Samples["validation"]
        if(DISetup.Name == "tttt"):
            print("The whole sample has {0} events".format(len(train.Events)))
            print("The training sample contains {0} events".format(len(train.Events)))
            Yield = np.sum(train.Weights)
        elif(DISetup.Name == "NLO"):
            print("The whole sample has {0} events".format(len(test.Events)+len(vali.Events)))
            print("The testing sample contains {0} events".format(len(test.Events)))
            print("The validation sample contains {0} events".format(len(vali.Events)))
            Yield = np.sum(test.Weights)+np.sum(vali.Weights)
        else:
            print("The whole sample has {0} events".format(len(train.Events)+len(test.Events)+len(vali.Events)))
            print("The training sample contains {0} events".format(len(train.Events)))
            print("The testing sample contains {0} events".format(len(test.Events)))
            print("The validation sample contains {0} events".format(len(vali.Events)))
            Yield = np.sum(train.Weights)+np.sum(test.Weights)+np.sum(vali.Weights)
        Utils.stdinfo("The total Yield amounts to: {0}".format(Yield))


    def ShuffleInUnison(self,DISample):
        assert len(DISample.Events) == len(DISample.Weights)
        assert len(DISample.Events) == len(DISample.OutTrue)
        permutation      = np.random.permutation(len(DISample.Events))
        shuffledEvents   = np.zeros(DISample.Events.shape, dtype=DISample.Events.dtype)
        shuffledWeights  = np.zeros(DISample.Weights.shape, dtype=DISample.Weights.dtype)
        shuffledOutTrue  = np.zeros(DISample.OutTrue.shape, dtype=DISample.OutTrue.dtype)
        for old_index, new_index in enumerate(permutation):
            shuffledEvents[new_index]   = DISample.Events[old_index]
            shuffledWeights[new_index]  = DISample.Weights[old_index]
            shuffledOutTrue[new_index]  = DISample.OutTrue[old_index]
        DISample.Events       = shuffledEvents
        DISample.Weights      = shuffledWeights
        DISample.OutTrue      = shuffledOutTrue
    
    def GetWeightSum(self, Weights,DISetup):
        """ Calculates the weights for each event from the weight expression """
        if(len(DISetup.WeightList) == 1):
            if(np.ndim(Weights) == 2):
                Weight = Weights.flatten()
            elif(np.ndim(Weights) == 3):
                Weight = Weights[:,:,0]
                Weight = Weight.flatten()
        else:
            SingleWeights = [(36207.7,284500),(44307.4,300000),(58450.1,310000)]
            if(len(DISetup.WeightList) == 9):
                Weight = Weights[:,0]*Weights[:,1]*Weights[:,2]*Weights[:,3]*Weights[:,4]*Weights[:,5]/Weights[:,6]*Weights[:,7]
            else:
                Weight = Weights[:,0]*Weights[:,1]*Weights[:,2]*Weights[:,3]*Weights[:,4]*Weights[:,5]

            
            for i in range(len(Weights)):
                for k in range(len(SingleWeights)):
                    if(Weights[i][DISetup.WeightList.index('runNumber')] == SingleWeights[k][1]):
                        Weight[i] *= SingleWeights[k][0]


        return Weight



def SaveNpy(fname,Sample):
    fpath = os.path.expanduser("~")
    if(fpath == '/jwd'):
         fpath = '/cephfs/user/s6nsschw'
    np.save(fname+'_Events.npy',Sample.Events)
    os.system("mv "+fname+"_Events.npy"+fpath+"/Data/Fast/")
    Other = np.hstack((Sample.Weights.reshape(-1,1),Sample.OutTrue.reshape(-1,1),Sample.MultiClass.reshape(-1,1)))
    np.save(fname+'_Other.npy',Other)
    os.system("mv "+fname+"_Other.npy"+fpath+"/Data/Fast/")







