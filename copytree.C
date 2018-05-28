void copytree()
{
	TString trname[132]={"wt_DR_nominal","wt_DS","tt_nominal","tt_JET_21NP_JET_EffectiveNP_1__1down","tt_JET_21NP_JET_EffectiveNP_1__1up"};

	TFile *f_all=new TFile("~/internship/data_single_lepton/reprocess_3j1b_v15_20180123.root");
	TFile *f_sub=new TFile("reprocess_3j1b_nominal_kirfel.root","RECREATE");

	for(int i=0;i<5;i++)
	//for(int i=0;i<2;i++)
	{

		TTree *tr_all=(TTree*)f_all->Get(trname[i]);
		TTree *tr_sub;//=new TTree("infmom");

		//tr_sub=tr_all->CopyTree("pass_1j1b&&pass_LeptonCut&&pass_3rdLeptonVeto&&pass_CMSMETRequirement&&pass_TruthMatch","",50,0);
		tr_sub=tr_all->CopyTree("pass_3j1b&&pass_TruthMatch","",1000000000000000,0);

		cout<<trname[i]<<endl;
		tr_sub->Write("", TObject::kOverwrite);
		delete tr_sub;
	}
	f_sub->Close();
}

void copytree2()
{
	TString trname[132]={"data_nominal","wt_DR_nominal","tt_nominal","wt_DS","tt_radHi", "tt_radLo",
"wt_DR_JET_21NP_JET_EffectiveNP_1__1down",
"tt_JET_21NP_JET_EffectiveNP_1__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_1__1up",
"tt_JET_21NP_JET_EffectiveNP_1__1up",
"wt_DR_JET_21NP_JET_BJES_Response__1down",
"tt_JET_21NP_JET_BJES_Response__1down",
"wt_DR_EG_SCALE_ALL__1down",
"tt_EG_SCALE_ALL__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_5__1down",
"tt_JET_21NP_JET_EffectiveNP_5__1down",
"wt_DR_MUON_MS__1down",
"tt_MUON_MS__1down",
"wt_DR_MUON_SAGITTA_RHO__1down",
"tt_MUON_SAGITTA_RHO__1down",
"wt_DR_MUON_SAGITTA_RHO__1up",
"tt_MUON_SAGITTA_RHO__1up",
"wt_af2_nominal",
"wt_af2_radHi",
"wt_af2_radLo",
"wt_af2_PS",
"wt_af2_ME",
"tt_af2_nominal",
"tt_af2_ME",
"tt_af2_PS",
"wt_DR_EG_RESOLUTION_ALL__1down",
"tt_EG_RESOLUTION_ALL__1down",
"wt_DR_JET_21NP_JET_Flavor_Response__1up",
"tt_JET_21NP_JET_Flavor_Response__1up",
"wt_DR_JET_21NP_JET_Pileup_OffsetMu__1down",
"tt_JET_21NP_JET_Pileup_OffsetMu__1down",
"wt_DR_JET_21NP_JET_Pileup_OffsetNPV__1down",
"tt_JET_21NP_JET_Pileup_OffsetNPV__1down",
"wt_DR_JET_21NP_JET_Pileup_OffsetMu__1up",
"tt_JET_21NP_JET_Pileup_OffsetMu__1up",
"wt_DR_JET_21NP_JET_PunchThrough_MC15__1down",
"tt_JET_21NP_JET_PunchThrough_MC15__1down",
"wt_DR_JET_JER_SINGLE_NP__1up",
"tt_JET_JER_SINGLE_NP__1up",
"wt_DR_MUON_ID__1up",
"tt_MUON_ID__1up",
"wt_DR_MUON_SCALE__1down",
"tt_MUON_SCALE__1down",
"wt_DR_MET_SoftTrk_ScaleDown",
"tt_MET_SoftTrk_ScaleDown",
"wt_DR_JET_21NP_JET_EffectiveNP_2__1down",
"tt_JET_21NP_JET_EffectiveNP_2__1down",
"wt_DR_JET_21NP_JET_Pileup_PtTerm__1up",
"tt_JET_21NP_JET_Pileup_PtTerm__1up",
"wt_DR_MUON_SAGITTA_RESBIAS__1up",
"tt_MUON_SAGITTA_RESBIAS__1up",
"wt_DR_JET_21NP_JET_PunchThrough_MC15__1up",
"tt_JET_21NP_JET_PunchThrough_MC15__1up",
"wt_DR_JET_21NP_JET_Pileup_RhoTopology__1down",
"tt_JET_21NP_JET_Pileup_RhoTopology__1down",
"wt_DR_JET_21NP_JET_Flavor_Composition__1down",
"tt_JET_21NP_JET_Flavor_Composition__1down",
"wt_DR_JET_21NP_JET_Flavor_Response__1down",
"tt_JET_21NP_JET_Flavor_Response__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_8restTerm__1down",
"tt_JET_21NP_JET_EffectiveNP_8restTerm__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_7__1up",
"tt_JET_21NP_JET_EffectiveNP_7__1up",
"wt_DR_JET_21NP_JET_SingleParticle_HighPt__1up",
"tt_JET_21NP_JET_SingleParticle_HighPt__1up",
"wt_DR_JET_21NP_JET_EffectiveNP_4__1up",
"tt_JET_21NP_JET_EffectiveNP_4__1up",
"wt_DR_MET_SoftTrk_ScaleUp",
"tt_MET_SoftTrk_ScaleUp",
"wt_DR_JET_21NP_JET_EffectiveNP_8restTerm__1up",
"tt_JET_21NP_JET_EffectiveNP_8restTerm__1up",
"wt_DR_EG_RESOLUTION_ALL__1up",
"tt_EG_RESOLUTION_ALL__1up",
"wt_DR_JET_21NP_JET_EtaIntercalibration_TotalStat__1up",
"tt_JET_21NP_JET_EtaIntercalibration_TotalStat__1up",
"wt_DR_MET_SoftTrk_ResoPara",
"tt_MET_SoftTrk_ResoPara",
"wt_DR_EG_SCALE_ALL__1up",
"tt_EG_SCALE_ALL__1up",
"wt_DR_JET_21NP_JET_EffectiveNP_7__1down",
"tt_JET_21NP_JET_EffectiveNP_7__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_5__1up",
"tt_JET_21NP_JET_EffectiveNP_5__1up",
"wt_DR_MUON_MS__1up",
"tt_MUON_MS__1up",
"wt_DR_MUON_SAGITTA_RESBIAS__1down",
"tt_MUON_SAGITTA_RESBIAS__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_2__1up",
"tt_JET_21NP_JET_EffectiveNP_2__1up",
"wt_DR_JET_21NP_JET_Pileup_PtTerm__1down",
"tt_JET_21NP_JET_Pileup_PtTerm__1down",
"wt_DR_JET_21NP_JET_Flavor_Composition__1up",
"tt_JET_21NP_JET_Flavor_Composition__1up",
"wt_DR_MET_SoftTrk_ResoPerp",
"tt_MET_SoftTrk_ResoPerp",
"wt_DR_JET_21NP_JET_EffectiveNP_6__1up",
"tt_JET_21NP_JET_EffectiveNP_6__1up",
"wt_DR_JET_21NP_JET_EffectiveNP_4__1down",
"tt_JET_21NP_JET_EffectiveNP_4__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_6__1down",
"tt_JET_21NP_JET_EffectiveNP_6__1down",
"wt_DR_JET_21NP_JET_EffectiveNP_3__1up",
"tt_JET_21NP_JET_EffectiveNP_3__1up",
"wt_DR_JET_21NP_JET_EffectiveNP_3__1down",
"tt_JET_21NP_JET_EffectiveNP_3__1down",
"wt_DR_MUON_SCALE__1up",
"tt_MUON_SCALE__1up",
"wt_DR_JET_21NP_JET_EtaIntercalibration_NonClosure__1up",
"tt_JET_21NP_JET_EtaIntercalibration_NonClosure__1up",
"wt_DR_JET_21NP_JET_BJES_Response__1up",
"tt_JET_21NP_JET_BJES_Response__1up",
"wt_DR_JET_21NP_JET_SingleParticle_HighPt__1down",
"tt_JET_21NP_JET_SingleParticle_HighPt__1down",
"wt_DR_JET_21NP_JET_Pileup_RhoTopology__1up",
"tt_JET_21NP_JET_Pileup_RhoTopology__1up",
"wt_DR_JET_21NP_JET_Pileup_OffsetNPV__1up",
"tt_JET_21NP_JET_Pileup_OffsetNPV__1up",
"wt_DR_MUON_ID__1down",
"tt_MUON_ID__1down",
"wt_DR_JET_21NP_JET_EtaIntercalibration_Modelling__1up",
"tt_JET_21NP_JET_EtaIntercalibration_Modelling__1up",
"wt_DR_JET_21NP_JET_EtaIntercalibration_Modelling__1down",
"tt_JET_21NP_JET_EtaIntercalibration_Modelling__1down",
"wt_DR_JET_21NP_JET_EtaIntercalibration_NonClosure__1down",
"tt_JET_21NP_JET_EtaIntercalibration_NonClosure__1down",
"wt_DR_JET_21NP_JET_EtaIntercalibration_TotalStat__1down",
"tt_JET_21NP_JET_EtaIntercalibration_TotalStat__1down" };

	TFile *f_all=new TFile("reprocess_1j1b_nominal.root");
	TFile *f_sub=new TFile("test_1j1b_nominal.root","RECREATE");

	for(int i=0;i<132;i++)
	{

		TTree *tr_all=(TTree*)f_all->Get(trname[i]);
		TTree *tr_sub;//=new TTree("infmom");

		tr_sub=tr_all->CopyTree("","",10,0);

		cout<<trname[i]<<endl;

	}
	delete f_all;
	f_sub->Write();
	f_sub->Close();

}

