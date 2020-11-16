import configparser
import os

Syspath = os.path.expanduser("~")

config = configparser.ConfigParser()

config['Variables'] = {
                        'FNN18': 'HT_jets_noleadjet, met_met, leading_jet_pT, leading_bjet_pT, lep_0_pt, lep_1_pt, lep_0_phi, nJets, jet_sum_mv2c10_Continuous, deltaR_lb_max, deltaR_ll_min, deltaR_ll_max, deltaR_bb_min, deltaR_ll_sum, deltaR_lb_min, deltaR_lj_min, jet_5_pt, jet_1_pt',
                        'LSTM': 'jet_sum_mv2c10_Continuous, met_phi, nJets, met_met, el_eta, el_phi, el_pt, mu_eta, mu_phi, mu_pt, jet_eta, jet_phi, jet_pt'}

config['Paths'] = {
                    'ROOTPath':Syspath+'/Data/nominal_variables_v4_bootstrap/',
                    'configPath':'./config/ConfigFiles/',
                    'SavePath':Syspath+'/Data/Fast/'
}

config['ImportInfo'] = {
                        'ROOTTree':'nominal_Loose',
                        'DataSetInfo':'DataSets.ini',
                        'Cut': "nBTags_MV2c10_77>=2 && nJets>=6 && HT_all>500000 && (SSee_passECIDS==1 || SSem_passECIDS==1 || SSmm==1 || eee_Zveto==1 || eem_Zveto==1 || emm_Zveto==1 || mmm_Zveto==1)"
}

with open('MainConfig.ini','w') as configfile:
    config.write(configfile)