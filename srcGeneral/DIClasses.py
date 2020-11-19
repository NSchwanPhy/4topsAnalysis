from os.path import expanduser
from Utils import stdwar, stderr, stdinfo
import numpy as np

class DIAnaSetup:

    def __init__(self,Path,Tree,McChannel,LVars,WeightList,Name,Cuts):
        self.Path            = Path
        self.Tree            = Tree
        self.McChannel       = McChannel
        self.LVars           = LVars
        self.WeightList      = WeightList
        self.Name            = Name
        self.Cuts            = Cuts
        self.Samples         = DISample.empty()

def Init(VarKind,Samples,Cuts=True):

    tree                  = 'nominal_Loose'
    McChanneltttt         = "mcChannelNumber==412115"                                             #LO 412115 NLO 412043
    McChannelNLO          = "mcChannelNumber==412043"
    McChannelttW          = ""
    McChannelttWW         = "mcChannelNumber==410081"
    McChannelttZ          = "(mcChannelNumber==410156||mcChannelNumber==410157||mcChannelNumber==410218||mcChannelNumber==410219||mcChannelNumber==410220||mcChannelNumber==410276||mcChannelNumber==410277||mcChannelNumber==410278)"
    McChannelttH          = "(mcChannelNumber==346345||mcChannelNumber==346344)"
    McChannelvjets        = ""
    McChannelvv           = ""
    McChannelsingletop    = "event_BkgCategory==0 && ((mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410560||mcChannelNumber==410408))"
    McChannelothers       = "((mcChannelNumber>=364242&&mcChannelNumber<=364249)||mcChannelNumber==342284||mcChannelNumber==342285||mcChannelNumber==304014)"
    McChannelttbar_Qmis   = "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&event_BkgCategory==1"
    McChannelttbar_CO     = "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&& (event_BkgCategory==2 || event_BkgCategory==3)"
    McChannelttbar_HF     = "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&(event_BkgCategory==4||(((mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&event_BkgCategory==0))"
    McChannelttbar_light  = "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&event_BkgCategory==5"
    McChannelttbar_others = "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&(event_BkgCategory==6)"
    McChannelTruth4t      = "mcChannelNumber==412043"
    McChannelTruth3t      = ""
 
    WListBkg           = ['weight_normalise', 'weight_pileup', 'weight_jvt', 'weight_leptonSF', 'weight_bTagSF_MV2c10_Continuous_CDI20190730',
                    'weight_mc','runNumber']
    WListSig           = ['weight_normalise', 'weight_pileup', 'weight_jvt', 'weight_leptonSF', 'weight_bTagSF_MV2c10_Continuous_CDI20190730',
                     'weight_mcweight_normalise[85]', 'weight_mcweight_normalise[0]', 'mc_generator_weights[85]','runNumber']
    WListNLO           = WListBkg
    WeightLO  = ["((weight_normalise*weight_mcweight_normalise[85]/weight_mcweight_normalise[0]*weight_pileup*weight_jvt*mc_generator_weights[85]*weight_leptonSF*weight_bTagSF_MV2c10_Continuous_CDI20190730)*(36207.7*(runNumber==284500)+44307.4*(runNumber==300000)+(runNumber==310000)*58450.1))"]
    WeightNLO = ["(36207.7*(runNumber==284500)+44307.4*(runNumber==300000)+(runNumber==310000)*58450.1)*weight_normalise*weight_pileup*weight_jvt*weight_mc*weight_leptonSF*weight_bTagSF_MV2c10_Continuous_CDI20190730"]

    CutsTruth = "nBTags_MV2c10_77>=2 && nJets>=8 && HT_all>500000 && (SSee_passECIDS==1 || SSem_passECIDS==1 || SSmm==1)"
    if(Cuts == False):
        Cuts = ''
    else:
        Cuts = "nBTags_MV2c10_77>=2 && nJets>=6 && HT_all>500000 && (SSee_passECIDS==1 || SSem_passECIDS==1 || SSmm==1 || eee_Zveto==1 || eem_Zveto==1 || emm_Zveto==1 || mmm_Zveto==1)"

    if(VarKind == 'FNN18'):
        listOfVariables = ['HT_jets_noleadjet', 'met_met', 'leading_jet_pT', 'leading_bjet_pT',
                            'lep_0_pt', 'lep_1_pt', 'lep_0_phi' , 'nJets', 'jet_sum_mv2c10_Continuous',
                            'deltaR_lb_max','deltaR_ll_min', 'deltaR_ll_max', 'deltaR_bb_min', 'deltaR_ll_sum',
                            'deltaR_lb_min','deltaR_lj_min', 'jet_5_pt' ,'jet_1_pt',]
    elif(VarKind == 'FNN12'):
        listOfVariables = ['jet_sum_mv2c10_Continuous', 'lepton_0_Pt', 'met_met','deltaR_ll_min', 'jet_5_pt',
                            'deltaR_lb_max', 'HT_jets_noleadjet','jet_1_pt',
                            'deltaR_ll_sum','leading_jet_pT', 'deltaR_bj_min', 'leading_bjet_pT']
    elif(VarKind == 'FNN5'):
        listOfVariables = ['jet_sum_mv2c10_Continuous', 'nJets', 'deltaR_ll_min', 'deltaR_ll_sum', 'met_met']
    elif(VarKind == 'LowLevel'):
        listOfVariables = ['test']
    elif(VarKind == 'LSTM'):
        listOfVariables = ['jet_sum_mv2c10_Continuous', 'met_phi', 'nJets', 'met_met', 'el_eta', 'el_phi','el_pt', 'mu_eta', 'mu_phi','mu_pt',
        'jet_eta', 'jet_phi','jet_pt']
    elif(VarKind == 'RNNLowLevel'):
        listOfVariables = ['nJets', 'jet_sum_mv2c10_Continuous', 'met_met', 'met_phi', 'el_eta', 'el_phi','el_pt', 'mu_eta', 'mu_phi','mu_pt',
        'jet_eta', 'jet_phi','jet_pt']
    elif(VarKind == 'NLO'):
        listOfVariables = ['jet_sum_mv2c10_Continuous', 'nJets', 'deltaR_ll_min', 'deltaR_ll_sum', 'met_met']
        McChanneltttt      = "mcChannelNumber==412043"
        WListSig = WListBkg
    elif(VarKind == 'CompareLow'):
        listOfVariables = ['Lepton_0_eta','Lepton_1_eta','Lepton_0_phi','Lepton_1_phi']
    elif(VarKind == 'CompareHigh'):
        listOfVariables = ['deltaR_ll_sum', 'Lepton_0_eta']
    elif(VarKind == 'Top Reco'):
        listOfVariables = ['jet_pt','jet_eta','jet_phi','jet_e','jet_mv2c10']
    elif(VarKind == 'AutoE'):
        listOfVariables  = ['nJets', 'met_met', 'met_phi', 'el_eta', 'el_phi','el_pt', 'mu_eta', 'mu_phi','mu_pt',
        'jet_eta', 'jet_phi','jet_pt']
    elif(VarKind == 'TruthMatch'):
        listOfVariables  = ['truth_tbar1_pt', 'truth_tbar1_eta', 'truth_tbar1_phi', 'truth_tbar1_e', 'truth_tbar1_isHad',
                            'truth_tbar2_pt', 'truth_tbar2_eta', 'truth_tbar2_phi', 'truth_tbar2_e', 'truth_tbar2_isHad',
                            'truth_top1_pt', 'truth_top1_eta', 'truth_top1_phi', 'truth_top1_e', 'truth_top1_isHad',
                            'truth_top2_pt', 'truth_top2_eta', 'truth_top2_phi', 'truth_top2_e', 'truth_top2_isHad']
    elif(VarKind == 'Childtbar1'):
        listOfVariables  = ['truth_tbar1_initialState_child0_pt', 'truth_tbar1_initialState_child0_eta', 'truth_tbar1_initialState_child0_phi', 'truth_tbar1_initialState_child0_e', 'truth_tbar1_initialState_child0_pdgid',
                            'truth_tbar1_initialState_child1_pt', 'truth_tbar1_initialState_child1_eta', 'truth_tbar1_initialState_child1_phi', 'truth_tbar1_initialState_child1_e', 'truth_tbar1_initialState_child1_pdgid',
                            'truth_tbar1_initialState_child2_pt', 'truth_tbar1_initialState_child2_eta', 'truth_tbar1_initialState_child2_phi', 'truth_tbar1_initialState_child2_e', 'truth_tbar1_initialState_child2_pdgid']
    elif(VarKind == 'Childtbar2'):
        listOfVariables  = ['truth_tbar2_initialState_child0_pt', 'truth_tbar2_initialState_child0_eta', 'truth_tbar2_initialState_child0_phi', 'truth_tbar2_initialState_child0_e', 'truth_tbar2_initialState_child0_pdgid',
                            'truth_tbar2_initialState_child1_pt', 'truth_tbar2_initialState_child1_eta', 'truth_tbar2_initialState_child1_phi', 'truth_tbar2_initialState_child1_e', 'truth_tbar2_initialState_child1_pdgid',
                            'truth_tbar2_initialState_child2_pt', 'truth_tbar2_initialState_child2_eta', 'truth_tbar2_initialState_child2_phi', 'truth_tbar2_initialState_child2_e', 'truth_tbar2_initialState_child2_pdgid']
    elif(VarKind == 'Childtop1'):
        listOfVariables  = ['truth_top1_initialState_child0_pt', 'truth_top1_initialState_child0_eta', 'truth_top1_initialState_child0_phi', 'truth_top1_initialState_child0_e', 'truth_top1_initialState_child0_pdgid',
                            'truth_top1_initialState_child1_pt', 'truth_top1_initialState_child1_eta', 'truth_top1_initialState_child1_phi', 'truth_top1_initialState_child1_e', 'truth_top1_initialState_child1_pdgid',
                            'truth_top1_initialState_child2_pt', 'truth_top1_initialState_child2_eta', 'truth_top1_initialState_child2_phi', 'truth_top1_initialState_child2_e', 'truth_top1_initialState_child2_pdgid']
    elif(VarKind == 'Childtop2'):
        listOfVariables  = ['truth_top2_initialState_child0_pt', 'truth_top2_initialState_child0_eta', 'truth_top2_initialState_child0_phi', 'truth_top2_initialState_child0_e', 'truth_top2_initialState_child0_pdgid',
                            'truth_top2_initialState_child1_pt', 'truth_top2_initialState_child1_eta', 'truth_top2_initialState_child1_phi', 'truth_top2_initialState_child1_e', 'truth_top2_initialState_child1_pdgid',
                            'truth_top2_initialState_child2_pt', 'truth_top2_initialState_child2_eta', 'truth_top2_initialState_child2_phi', 'truth_top2_initialState_child2_e', 'truth_top2_initialState_child2_pdgid']
    elif(VarKind == 'BasicTruth'):
        listOfVariables = ['truth_nVectorBoson','el_isTight','mu_isTight','jet_pt','jet_eta','jet_phi','jet_e','jet_isbtagged_MV2c10_77','el_pt','el_eta','el_phi','el_e','mu_pt','mu_eta','mu_phi','mu_e']
    elif(VarKind == 'HadTopDiscrim'):
        listOfVariables = ['truth_tbar1_isHad','truth_tbar2_isHad','truth_top1_isHad','truth_top2_isHad','met_met','met_phi','el_isTight','mu_isTight','nJets','jet_mv2c10','jet_eta','jet_phi','jet_pt',
                           'mu_eta','mu_phi','el_eta','el_phi','nBTags_MV2c10_77']
    elif(VarKind == 'ThreevsFour'):
        listOfVariables = ['el_isTight','mu_isTight','jet_pt','jet_eta','jet_phi','jet_e','el_pt','el_eta','el_phi','el_e','mu_pt','mu_eta','mu_phi','mu_e','met_met','met_phi']
    else:
        listOfVariables = VarKind


    if(expanduser("~") == '/gpfs/share/home/s6nsschw'):
        FilePath = '/cephfs/user/s6nsschw/Data/nominal_variables_v4_bootstrap/'
    elif(expanduser("~") == '/home/nschwan'):
        FilePath = '~/Data/nominal_variables_v4_bootstrap/'

    elif(expanduser("~") == '/home/niklas'):
        FilePath = '~/Data/nominal_variables_v4_bootstrap/'
    else:
        FilePath = "/cephfs/user/s6nsschw/Data/nominal_variables_v4_bootstrap/"
        stdwar("This system is not knewn")

    pathtttt        = [FilePath+'mc16a/2lss3lge1mv2c10j/4tops.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/4tops.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/4tops.root']

    pathttW         = [FilePath+'mc16a/2lss3lge1mv2c10j/ttWSherpa.root',    #Sherpa
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttWSherpa.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttWSherpa.root']

    pathttWW         = [FilePath+'mc16a/2lss3lge1mv2c10j/ttWW.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttWW.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttWW.root'] 

    pathttZ         = [FilePath+'mc16a/2lss3lge1mv2c10j/ttZ.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttZ.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttZ.root',
                        FilePath+'mc16a/2lss3lge1mv2c10j/ttll.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttll.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttll.root']

    pathttH         = [FilePath+'mc16a/2lss3lge1mv2c10j/ttH.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttH.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttH.root']

    pathvjets       = [FilePath+'mc16a/2lss3lge1mv2c10j/vjets.root',   
                        FilePath+'mc16d/2lss3lge1mv2c10j/vjets.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/vjets.root']

    pathvv          = [FilePath+'mc16a/2lss3lge1mv2c10j/vv.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/vv.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/vv.root']

    pathsingletop   = [FilePath+'mc16a/2lss3lge1mv2c10j/single-top.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/single-top.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/single-top.root'] 

    pathothers      = [FilePath+'mc16a/2lss3lge1mv2c10j/ttt.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttt.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttt.root',
                        FilePath+'mc16a/2lss3lge1mv2c10j/vv.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/vv.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/vv.root',
                        FilePath+'mc16a/2lss3lge1mv2c10j/vvv.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/vvv.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/vvv.root',
                        FilePath+'mc16a/2lss3lge1mv2c10j/vh.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/vh.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/vh.root']

    pathttbar_Else  = [FilePath+'mc16a/2lss3lge1mv2c10j/ttbar.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/ttbar.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/ttbar.root',
                        FilePath+'mc16a/2lss3lge1mv2c10j/single-top.root',
                        FilePath+'mc16d/2lss3lge1mv2c10j/single-top.root',
                        FilePath+'mc16e/2lss3lge1mv2c10j/single-top.root']

    # pathTruth4tops  = [FilePath+'truthsamples/old/4tops_truth.root']
    # pathTruth3tops  = [FilePath+'truthsamples/old/3tops_truth.root']

    pathTruth4tops  = [FilePath+'truthsamples/4tops_mc16e.root']
    pathTruth3tops  = [FilePath+'truthsamples/3tops_mc16e.root']                                                                        


    NLOSample           = DIAnaSetup(pathtttt,tree,McChannelNLO,listOfVariables,WeightNLO,'NLO',Cuts)
    ttttSample          = DIAnaSetup(pathtttt,tree,McChanneltttt,listOfVariables,WeightLO,'tttt',Cuts)
    ttWSample           = DIAnaSetup(pathttW,tree,McChannelttW,listOfVariables,WeightNLO,'ttW',Cuts)
    ttWWSample          = DIAnaSetup(pathttWW,tree,McChannelttWW,listOfVariables,WeightNLO,'ttWW',Cuts)
    ttZSample           = DIAnaSetup(pathttZ,tree,McChannelttZ,listOfVariables,WeightNLO,'ttZ',Cuts)
    ttHSample           = DIAnaSetup(pathttH,tree,McChannelttH,listOfVariables,WeightNLO,'ttH',Cuts)
    vjetsSample         = DIAnaSetup(pathvjets,tree,McChannelvjets,listOfVariables,WeightNLO,'vjets',Cuts)
    vvSample            = DIAnaSetup(pathvv,tree,McChannelvv,listOfVariables,WeightNLO,'vv',Cuts)
    singletopSample     = DIAnaSetup(pathsingletop,tree,McChannelsingletop,listOfVariables,WeightNLO,'singletop',Cuts)
    othersSample        = DIAnaSetup(pathothers,tree,McChannelothers,listOfVariables,WeightNLO,'others',Cuts)
    ttbar_QmisSample    = DIAnaSetup(pathttbar_Else,tree,McChannelttbar_Qmis,listOfVariables,WeightNLO,'ttbar Qmis',Cuts)
    ttbar_COSample      = DIAnaSetup(pathttbar_Else,tree,McChannelttbar_CO,listOfVariables,WeightNLO,'ttbar CO',Cuts)
    ttbar_HFSample      = DIAnaSetup(pathttbar_Else,tree,McChannelttbar_HF,listOfVariables,WeightNLO,'ttbar HF',Cuts)
    ttbar_lightSample   = DIAnaSetup(pathttbar_Else,tree,McChannelttbar_light,listOfVariables,WeightNLO,'ttbar light',Cuts)
    ttbar_othersSample  = DIAnaSetup(pathttbar_Else,tree,McChannelttbar_others,listOfVariables,WeightNLO,'ttbar others',Cuts)
    Truth4tops          = DIAnaSetup(pathTruth4tops,tree,McChannelTruth4t,listOfVariables,WeightNLO,'Truth4tops',CutsTruth)
    Truth3tops          = DIAnaSetup(pathTruth3tops,tree,McChannelTruth3t,listOfVariables,WeightNLO,'Truth3tops',CutsTruth)



    LSamples = [ttttSample,NLOSample,othersSample,vjetsSample,vvSample,singletopSample,ttbar_othersSample,ttbar_lightSample,
            ttbar_HFSample,ttbar_COSample,ttbar_QmisSample,ttHSample,ttZSample,ttWWSample,ttWSample,Truth4tops,Truth3tops]

    if(Samples == 'All'):
        Samples = LSamples
    elif(Samples == 'nomLoose'):
        Samples = LSamples = [ttttSample,NLOSample,othersSample,vjetsSample,vvSample,singletopSample,ttbar_othersSample,ttbar_lightSample,
            ttbar_HFSample,ttbar_COSample,ttbar_QmisSample,ttHSample,ttZSample,ttWWSample,ttWSample]
    else:
        Samples = [s for s in LSamples if s.Name in Samples]

    return Samples

""" ------------------------------------------------------------------------------------------------------------------------------------------------ """


class DISample:

    def __init__(self,Events,Weights,OutTrue,MultiClass,LVariables,Names):
        self.Events       = Events
        self.Weights      = Weights
        self.OutTrue      = OutTrue
        self.MultiClass   = MultiClass
        self.LVariables   = LVariables
        self.Names        = Names

    @classmethod
    def empty(cls):
        """ empty instance """
        return cls(None,None,None,None,None,None)


""" ------------------------------------------------------------------------------------------------------------------------------------------------ """


class DILrSchedule:
    def __init__(self,mode,Lr,factor=1.,cycle='triangular',MinLr=0.006,StepSize=0):
        self.mode = mode            # cycle,poly,normal,drop
        self.Lr = Lr                # intial learnrate or max learnrate (cycle)
        self.factor = factor        # poly => power, drop => factor, cycle => gamma (used in exp_range)
        self.cycle = cycle          # mode of the cycle scheduel {triangular, triangular2, exp_range}
        self.MinLr = MinLr          # cycle only
        self.StepSize = StepSize    # drop => dropEvery, cycle => StepSize, poly => Epochs

    def Print(self):
        if(self.mode == 'normal'):
            print("Learn rate: {0}".format(self.Lr))
        elif(self.mode == 'drop'):
            print("Inital learn rate: {0}".format(self.Lr))
            print("Drop of Every: {0}".format(self.StepSize))
            print("Factor: {0}".format(self.factor))
        elif(self.mode == 'poly'):
            print("Inital learn rate: {0}".format(self.Lr))
            print("Power: {0}".format(self.factor))
            print("Max Epochs: {0}".format(self.StepSize))
        elif(self.mode == 'cycle'):
            print("Mode: {0}".format(self.cycle))
            print("Max learn rate: {0}".format(self.Lr))
            print("Base learn rate: {0}".format(self.MinLr))
            print("Step size: {0}".format(self.StepSize))
            if(self.cycle == 'exp_range'):
                print("Gamma: {0}".format(self.factor))

""" ------------------------------------------------------------------------------------------------------------------------------------------------ """


class DIANNSetup:

    def __init__(self,Architecture,Epochs,SavePath,Batch,ModelName,Neurons,
                 InputDim=5,Optimizer='Adam',LearnRate=DILrSchedule('normal',0.001),
                 Dropout=None,Regu=0,Winit='GlorotNormal',Activ='relu'):
        self.Architecture   = Architecture                           #Which NN architecture
        self.Epochs         = str(Epochs)
        self.SavePath       = SavePath
        self.Batch          = str(Batch)
        self.ModelName      = ModelName
        self.InputDim       = InputDim
        self.Optimizer      = Optimizer
        self.Neurons        = Neurons
        self.LearnRate      = LearnRate
        self.Regu           = Regu
        self.Winit          = Winit
        self.Activ          = Activ
        if(Dropout == None and Architecture == 'LSTM'):
            self.Dropout = [0 for layer in Neurons[0]]
        else:
            self.Dropout        = Dropout

""" ------------------------------------------------------------------------------------------------------------------------------------------------ """


class DIBDTSetup:

    def __init__(self,ModelName,TreeNumber,MaxDepth,NMinActual,Shrinkage,NCutActual,BaggingActual):
        self.ModelName      = ModelName
        self.TreeNumber     = str(TreeNumber)
        self.MaxDepth       = str(MaxDepth)
        self.NMinActual     = str(NMinActual)
        self.NCutActual     = str(NCutActual)
        self.BaggingActual  = '0.'+str(BaggingActual)
        if(Shrinkage == 1):
            self.Shrinkage = '0.01'
        elif(Shrinkage == 2):
            self.Shrinkage = '0.02'
        elif(Shrinkage == 3):
            self.Shrinkage = '0.05'
        elif(Shrinkage == 4):
            self.Shrinkage = '0.1'
        else:
            self.Shrinkage = '0.2'

""" ------------------------------------------------------------------------------------------------------------------------------------------------ """


class DIEvalution:

    def __init__(self,SavePath,ModelNames,DataSet,ModelEven=None,ModelOdd=None,HistoryEven=None,HistoryOdd=None):
        if(isinstance(ModelNames,str)):
            ModelNames = [ModelNames]

        self.SavePath    = SavePath
        self.ModelNames  = ModelNames
        self.DataSet     = DataSet
        self.ModelEven   = ModelOdd
        self.ModelOdd    = ModelOdd
        self.HistoryEven = HistoryEven
        self.HistoryOdd  = HistoryOdd

""" ------------------------------------------------------------------------------------------------------------------------------------------------ """


class DIDataSet:

    def __init__(self,train,test,vali):
        self.LO         = train.Events[train.OutTrue == 1]
        self.LOW        = train.Weights[train.OutTrue == 1]
        self.NLO        = test.Events[test.OutTrue == 1]
        self.NLOW       = test.Weights[test.OutTrue == 1]
        self.EvenBkg    = train.Events[train.OutTrue == 0]
        self.EvenBkgW   = train.Weights[train.OutTrue == 0]
        self.EvenMulti  = train.MultiClass[train.OutTrue == 0]
        self.OddBkg     = test.Events[test.OutTrue == 0]
        self.OddBkgW    = test.Weights[test.OutTrue == 0]
        self.OddMulti   = test.MultiClass[test.OutTrue == 0]
        self.vali       = vali
        self.LVariables = vali.LVariables
        self.Names      = vali.Names

    def GetInput(self, ModelName):
        #validation in both cases is 20% Bkg and 20% NLO
        if("Even" in ModelName):
            #train in the Even case is 40% Even Bkg + 100% LO
            Events      = np.vstack((self.LO,self.EvenBkg))
            Weights     = np.append(self.LOW,self.EvenBkgW)                                
            OutTrue     = np.append(np.ones(len(self.LO)),np.zeros(len(self.EvenBkg)))
            MultiClass  = np.append(np.zeros(len(self.LO)), self.EvenMulti) 
            train = DISample(Events,Weights,OutTrue,MultiClass,self.LVariables,self.Names)
            #test in the Even case is 40% Odd Bkg + 80% NLO
            Events      = np.vstack((self.NLO,self.OddBkg))
            Weights     = np.append(self.NLOW,self.OddBkgW)
            OutTrue     = np.append(np.ones(len(self.NLO)),np.zeros(len(self.OddBkg)))
            MultiClass  = np.append(np.zeros(len(self.NLO)), self.OddMulti)
            test = DISample(Events,Weights,OutTrue,MultiClass,self.LVariables,self.Names)
            return train, test

        elif("Odd" in ModelName):
            #train in the Odd case is 40% Odd Bkg + 100% LO
            Events      = np.vstack((self.LO,self.OddBkg))
            Weights     = np.append(self.LOW,self.OddBkgW)
            OutTrue     = np.append(np.ones(len(self.LO)),np.zeros(len(self.OddBkg)))
            MultiClass  = np.append(np.zeros(len(self.LO)), self.OddMulti) 
            train = DISample(Events,Weights,OutTrue,MultiClass,self.LVariables,self.Names)
            #test in the Odd case is 40% Even Bkg + 80% NLO
            Events      = np.vstack((self.NLO,self.EvenBkg))
            Weights     = np.append(self.NLOW,self.EvenBkgW)
            OutTrue     = np.append(np.ones(len(self.NLO)),np.zeros(len(self.EvenBkg)))
            MultiClass  = np.append(np.zeros(len(self.NLO)), self.EvenMulti)
            test = DISample(Events,Weights,OutTrue,MultiClass,self.LVariables,self.Names)
            return train, test

        else:
            stderr("The ModelName does not include Even or Odd")
            assert 0 == 1

    def OneSample(self,key):
        #Returns all events in one Sample
        #key either NLO or LO
        if(key == 'NLO'):
            Sig  = self.NLO
            SigW = self.NLOW
        elif(key == 'LO'):
            Sig  = self.LO
            SigW = self.LOW

        Events = np.append(Sig,self.EvenBkg)
        Events = np.append(Events,self.OddBkg)

        Weights = np.append(SigW,self.EvenBkgW)
        Weights = np.append(SigW,self.OddBkgW)

        OutTrue = np.append(np.ones(len(Sig)),np.zeros(len(self.EvenBkgW)+len(self.OddBkg)))
        MultiClass = np.append(np.ones(len(Sig)),self.EvenMulti)
        MultiClass = np.append(MultiClass,self.OddMulti)

        return DISample(Events,Weights,OutTrue,MultiClass,self.LVariables,self.Names)



        

    








