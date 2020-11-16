import configparser

""" Creates the config file that includes the information about the DataSets.
    Monte Carlo channel nummber, path to root files, sum of weights, """


config = configparser.ConfigParser()


DataSets = ['ttttLO','ttttNLO','ttW','ttZ',
            'ttH','ttCO','ttHF','ttQmisID',
            'ttWW','ttlight','ttother','singletop',
            'VV','others','Vjets']

Files = [['4tops'],['4tops'],['ttWSherpa'],['ttZ','ttll'],
        ['ttH'],['ttbar','single-top'],['ttbar','single-top'],['ttbar','single-top'],
        ['ttWW'],['ttbar','single-top'],['ttbar','single-top'],['single-top'],
        ['vv'],['ttt','vv','vvv','vh'],['vjets']]



config['McChannel'] =   {'ttttLO':"mcChannelNumber==412115",
                        'ttttNLO': "mcChannelNumber==412043",
                        'ttW':"",
                        'ttZ':"(mcChannelNumber==410156||mcChannelNumber==410157||mcChannelNumber==410218||mcChannelNumber==410219||mcChannelNumber==410220||mcChannelNumber==410276||mcChannelNumber==410277||mcChannelNumber==410278)",
                        'ttH': "(mcChannelNumber==346345||mcChannelNumber==346344)",
                        'ttCO': "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&& (event_BkgCategory==2 || event_BkgCategory==3)",
                        'ttHF': "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&(event_BkgCategory==4||(((mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&event_BkgCategory==0))",
                        'ttQmisID': "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&event_BkgCategory==1",
                        'ttWW': "mcChannelNumber==410081",
                        'ttlight': "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&event_BkgCategory==5",
                        'ttother': "((mcChannelNumber==410658||mcChannelNumber==410659||mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410644||mcChannelNumber==410645)||(mcChannelNumber>=407342&&mcChannelNumber<=407344)||(((mcChannelNumber==410470))&&GenFiltHT/1000.<600))&&(event_BkgCategory==6)",
                        'singletop': "event_BkgCategory==0 && ((mcChannelNumber==410646||mcChannelNumber==410647||mcChannelNumber==410560||mcChannelNumber==410408))",
                        'VV': "",
                        'others': "((mcChannelNumber>=364242&&mcChannelNumber<=364249)||mcChannelNumber==342284||mcChannelNumber==342285||mcChannelNumber==304014)",
                        'Vjets': ""
}


#Weight expressions (all samples are NLO, besides ttttLO)
WeightLO  = "((weight_normalise*weight_mcweight_normalise[85]/weight_mcweight_normalise[0]*weight_pileup*weight_jvt*mc_generator_weights[85]*weight_leptonSF*weight_bTagSF_MV2c10_Continuous_CDI20190730)*(36207.7*(runNumber==284500)+44307.4*(runNumber==300000)+(runNumber==310000)*58450.1))"
WeightNLO = "(36207.7*(runNumber==284500)+44307.4*(runNumber==300000)+(runNumber==310000)*58450.1)*weight_normalise*weight_pileup*weight_jvt*weight_mc*weight_leptonSF*weight_bTagSF_MV2c10_Continuous_CDI20190730"
config['Weights'] = {}
for DataSet in DataSets:
    if(DataSet != 'ttttLO'):
        config['Weights'][DataSet] = WeightNLO
    else:
        config['Weights'][DataSet] = WeightLO



for i,DataSet in enumerate(DataSets):
    config['Files '+DataSet] = {}
    n = 0
    for File in Files[i]:
        config['Files '+DataSet]['File'+str(n)] = 'mc16a/2lss3lge1mv2c10j/'+File+'.root'
        n+=1
        config['Files '+DataSet]['File'+str(n)] = 'mc16d/2lss3lge1mv2c10j/'+File+'.root'
        n+=1
        config['Files '+DataSet]['File'+str(n)] = 'mc16e/2lss3lge1mv2c10j/'+File+'.root'
        n+=1


with open('DataSets.ini','w') as configfile:
    config.write(configfile)