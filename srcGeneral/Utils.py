import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def stderr(message):
    print(FAIL + message + ENDC)

def stdwar(message):
    print(WARNING + message + ENDC)

def stdinfo(message):
    print(OKBLUE + message + ENDC)

def dlen(obj):
    """ returns the length of the obj if the obj is an int or a float len=1 """
    if(isinstance(obj,list)):
        Length = len(obj)
    elif(isinstance(obj,np.ndarray)):
        Length = len(obj)
    elif(isinstance(obj,int) or isinstance(obj,float)):
        Length = 1
    else:
        stderr("object type not supported")
        print(obj, type(obj))
        assert 0 == 1

    return Length

def ContainsList(Arr):
    """ If array containce list True if not Flase """
    Flag = False
    for element in Arr:
        if(isinstance(element,list)):
            Flag = True
    return Flag

def PadArr(obj,MaxLength,PadNum=-3.4*pow(10,38)):
    """ Pads an the obj given to an MaxLength long list """
    if(hasattr(obj,'__len__')):
        if(isinstance(obj,np.ndarray)):
            obj = obj.tolist()
        diff = MaxLength - len(obj)
        if(diff > 0):
            for i in range(diff):
                obj.append(PadNum)
        elif(diff < 0):
                obj = obj[:MaxLength]
    else:
        obj = [obj]
        for i in range(MaxLength-1):
            obj.append(PadNum)
    
    return obj

def HasSequence(Arr):
    for obj in Arr[0]:
        if(isinstance(obj,np.ndarray)):
            print(obj,isinstance(obj,np.ndarray))
            return True
    return False

def Transform(Arr,kind):
    
    if(kind == 'ZScore'):  
        print("enter")           
        scaler = StandardScaler(with_mean=True, with_std=True)
        if(np.ndim(Arr) == 1):
            Arr = Arr.reshape(-1,1)
        scaler.fit(Arr)
        Arr = scaler.transform(Arr)
        Arr = Arr.ravel()
        
    if(kind == 'MinMax'):
        scaler = MinMaxScaler()
        if(np.ndim(Arr) == 1):
            Arr = Arr.reshape(-1,1)
        scaler.fit(Arr)
        Arr = scaler.transform(Arr)
        Arr = Arr.ravel()

    return Arr

def ConvertArr(Arr,SequenceLength,PadNum=-3.4*pow(10,38)):
    """ Converts the input array into an 3dim array with constant Sequence length (using padding)"""
    if(SequenceLength == 1):
        for irow in range(Arr.shape[0]):
            for icol in range(Arr.shape[1]):
                if(hasattr(Arr[irow][icol],'__len__')):
                    if(len(Arr[irow][icol]) == 0):
                        Arr[irow][icol] = PadNum
                    else:
                        Arr[irow][icol] = Arr[irow][icol][0]

    else:
        Batchs = []
        for j,Batch in enumerate(Arr):                               #Event loop
            for i, element in enumerate(Batch):         #Variable loop
                element = PadArr(element,SequenceLength,PadNum=PadNum)
                if(i == 0):
                    BatchArr = element
                else:
                    BatchArr = np.vstack((BatchArr,element))
            Batchs.append(BatchArr)
        Arr = np.array(Batchs)
    
    return Arr

def CheckSame(Arr1,Arr2):
    if(len(Arr1) != len(Arr2)):
        stdinfo("Not the same (length)")
    else:
        Truth = Arr1 == Arr2
        if(False in Truth):
            #stdinfo("Not the same")
            return False
        else:
            #stdinfo("They are the same")
            return True

def GetSamples(SampleAndSeed,DataSet,Name):
    Sample = SampleAndSeed[0]
    Seed   = SampleAndSeed[1]
    train, test = DataSet.GetInput(Name)
    vali = DataSet.vali
    if(Seed != None):
        Samples = {'train':train,'test':test,'vali':vali}
        Sample = Samples[Sample]
        
        Comb = np.hstack((Sample.Events,Sample.Weights.reshape(-1,1),Sample.OutTrue.reshape(-1,1)))
        Comb = resample(Comb, replace=True, n_samples=len(Comb), random_state=Seed)
        Sample.Events = Comb[:,:-2]
        Sample.Weights = Comb[:,-2:-1].ravel()
        Sample.OutTrue = np.array(Comb[:,-1], dtype=int)

    return train, test, vali




