void edit_variables()
{	//List of trees you want to keep
	TString trname[132]={"wt_DR_nominal","wt_DS","tt_nominal","tt_JET_21NP_JET_EffectiveNP_1__1down","tt_JET_21NP_JET_EffectiveNP_1__1up"};

	TFile *f_all=new TFile("~/internship/data_single_lepton/reprocess_3j1b_v15_20180123.root");//input
	TFile *f_sub=new TFile("reprocess_3j1b_nominal_kirfel_testedit.root","RECREATE");//output
	
	//input variables
	float Abs_Eta_Jet1 = 0;
	float Abs_Eta_Jet2 = 0;
	float Abs_Eta_Jet3 = 0;
	float Abs_Eta_Lep1 = 0;
	
	
	float eta_Jet1_address = 0;
	float eta_Jet2_address = 0;
	float eta_Jet3_address = 0;
	float eta_Lep1_address = 0;

	for(int i=0;i<5;i++)
	{

		TTree *tr_all=(TTree*)f_all->Get(trname[i]);
		TTree *tr_sub;//=new TTree("infmom");

		tr_sub=tr_all->CopyTree("pass_3j1b&&pass_TruthMatch");
		//CloneTree(0)
		
		TBranch *bnew1 = tr_sub->Branch("absEta_Jet1", &Abs_Eta_Jet1, "absEta_Jet1/F");
		TBranch *bnew2 = tr_sub->Branch("absEta_Jet2", &Abs_Eta_Jet2, "absEta_Jet2/F");
		TBranch *bnew3 = tr_sub->Branch("absEta_Jet3", &Abs_Eta_Jet3, "absEta_Jet3/F");
		TBranch *bnew4 = tr_sub->Branch("absEta_Lep1", &Abs_Eta_Lep1, "absEta_Lep1/F");
		
		tr_sub->SetBranchStatus("*",0);
		
		
		//new branches
		tr_sub->SetBranchStatus("absEta_Jet1",1);
		tr_sub->SetBranchStatus("absEta_Jet2",1);
		tr_sub->SetBranchStatus("absEta_Jet3",1);
		tr_sub->SetBranchStatus("absEta_Lep1",1);
		
		//old branches needed for computation
		tr_sub->SetBranchStatus("eta_Jet1",1);
		tr_sub->SetBranchStatus("eta_Jet2",1);
		tr_sub->SetBranchStatus("eta_Jet3",1);
		tr_sub->SetBranchStatus("eta_Lep1",1);
		
		
		//test to access variables
		tr_sub->SetBranchAddress("eta_Jet1", &eta_Jet1_address);
		tr_sub->SetBranchAddress("eta_Jet2", &eta_Jet2_address);
		tr_sub->SetBranchAddress("eta_Jet3", &eta_Jet3_address);
		tr_sub->SetBranchAddress("eta_Lep1", &eta_Lep1_address);
		
		Int_t nentries = tr_sub->GetEntries();
		for (Int_t event_counter=0;event_counter<nentries;event_counter++) {
		tr_sub->GetEvent(event_counter);
		
		
		//Compute new variables
		Abs_Eta_Jet1 = fabs( eta_Jet1_address );
		Abs_Eta_Jet2 = fabs( eta_Jet2_address );
		Abs_Eta_Jet3 = fabs( eta_Jet3_address );
		Abs_Eta_Lep1 = fabs( eta_Lep1_address );
		
		//Fill branches
		bnew1->Fill();
		bnew2->Fill();
		bnew3->Fill();
		bnew4->Fill();
		
		}
		
		tr_sub->SetBranchStatus("*",1);//reactivate
		cout<<trname[i]<<endl;
		tr_sub->Write("", TObject::kOverwrite);
		delete tr_sub;
	}
	f_sub->Close();
}
