#include <iostream>
#include <algorithm>
#include <fstream>
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
using namespace std;

void SlimDownTraining(){
	system("mkdir -vp data");
	system("mkdir -vp MC");
	system("mkdir -vp MC_training");

	vector<TString> trainingEvents = {"wt_DR_nominal","wt_DS","tt_nominal", \
			"wjetsL_nominal", "wjetsB_nominal", "wjetsC_nominal", "wjetsU_nominal", \
			"zjetsL_nominal", "zjetsB_nominal", "zjetsC_nominal", "zjetsU_nominal", \
			"diboson_nominal", "tZ_nominal", "schan_nominal", "tchan_nominal", "data_nominal" , \
			"fake_nominal"};

	vector<TString> testingEvents = {"wt_DR_nominal","wt_DS","tt_nominal", \
			"wjetsL_nominal", "wjetsB_nominal", "wjetsC_nominal", "wjetsU_nominal", \
			"zjetsL_nominal", "zjetsB_nominal", "zjetsC_nominal", "zjetsU_nominal", \
			"diboson_nominal", "tZ_nominal", "schan_nominal", "tchan_nominal", "data_nominal" , \
			"fake_nominal"};

	vector<TString> cutVariables = {"pt_Lep1", "pt_Jet1", "pt_Jet2", "pt_Jet3", \
			"MissingET", "massTrans_Lep1MET", "pass_3j1b", "pass_LeptonCut", "mass_WlWhBJet", "mass_pseudoWHad"};


	bool training;
	TString filename;
	TString outfilename;
	TString configPath = ;
	ifstream inFile;
	
	inFile.open(configPath + );
	if (!inFile) { //check if var list exists
		cerr << "Unable to open file datafile.txt" << endl;
		exit(1);
	}
	
	
	TString tmpString = "";
	int tmpInt = 0;
	float tmpEta, etaJet1;
	vector<TString> trainingVariables;
	
		while(inFile >> tmpString >> tmpInt){
		trainingVariables.push_back(tmpString);
	}
	
	TFile *oldfile = new TFile("reprocess_3j1b_nominal.root");
	
		for (TString event : testingEvents) {
		cout << "\nSlimming down " << event << endl;
		//check if current event is for training as well and define output training file
		if(std::find(trainingEvents.begin(), trainingEvents.end(), event) != trainingEvents.end()) {
			cout << "Will train on this channel" << endl;
			training = true;
		} else {training = false;}
		//define output file
		if(event.Contains("data")){
			outfilename="data/reprocessNB_3j1b_"+event+".root";
		} else {
			outfilename="MC/reprocessNB_3j1b_"+event+".root";
		}
		
		
				//get tree of the channel to be split and deactivate all branches but event weight
		TTree *oldtree = (TTree*)oldfile->Get(event);
		oldtree->SetName("nominal");
		oldtree->SetBranchStatus("*",0);
		oldtree->SetBranchStatus("EventWeight",1);
		oldtree->SetBranchStatus("flavour_Lep1",1);

		//only activate branches that are in the variable list
		TString tmpString = "";
		int tmpInt = 0;
		float tmpEta, etaJet1;
		for (auto trainingVariable: trainingVariables){
			if(trainingVariable.Contains("absEta_")){
				oldtree->SetBranchStatus("eta_Jet1",1);
				oldtree->SetBranchAddress("eta_Jet1", &tmpEta);
			} else {
				oldtree->SetBranchStatus(trainingVariable,1);
			}
		}

		//activate the branches for the cuts and assign values for per event evaluation
		float pt_Lep1, pt_Jet1, pt_Jet2, pt_Jet3, MissingET, massTrans_Lep1MET, mass_WlWhBJet, mass_pseudoWHad;
		bool pass_3j1b, pass_LeptonCut;
		for (auto cutVar: cutVariables){
			oldtree->SetBranchStatus(cutVar,1);
		}
		oldtree->SetBranchAddress("pt_Lep1", &pt_Lep1);
		oldtree->SetBranchAddress("pt_Jet1", &pt_Jet1);
		oldtree->SetBranchAddress("pt_Jet2", &pt_Jet2);
		oldtree->SetBranchAddress("pt_Jet3", &pt_Jet3);
		oldtree->SetBranchAddress("MissingET", &MissingET);
		oldtree->SetBranchAddress("massTrans_Lep1MET", &massTrans_Lep1MET);
		oldtree->SetBranchAddress("pass_3j1b", &pass_3j1b);
		oldtree->SetBranchAddress("pass_LeptonCut", &pass_LeptonCut);
		oldtree->SetBranchAddress("mass_WlWhBJet", &mass_WlWhBJet);
		oldtree->SetBranchAddress("mass_pseudoWHad", &mass_pseudoWHad);

		//clone to new trees but add no events for event by event testing
		//since I know I'll only need eta of the leading jet I'll just add a branch for the conversion to abs val
		TFile* outfile = new TFile(outfilename,"recreate");
		outfile->cd();
		TTree *newtree = oldtree->CloneTree(0);
		newtree->Branch("absEta_Jet1", &etaJet1, "absEta_Jet1/F");

		//need to create training file for all but will only fill for channels needed for training
		TFile* outfile_training = new TFile("MC_training/reprocessNB_3j1b_"+event+".root","recreate");
		TTree *newtree_training = oldtree->CloneTree(0);
		newtree_training->Branch("absEta_Jet1", &etaJet1, "absEta_Jet1/F");

		//event by event testing
		Long64_t nentries = oldtree->GetEntries();
		for (Long64_t i=0;i<nentries; i++) {
			oldtree->GetEntry(i);
			etaJet1 = fabs(tmpEta);
			if (pt_Lep1>27&&pt_Jet1>40&&pt_Jet2>30&&pt_Jet3>25&&MissingET>30&&\
					massTrans_Lep1MET>40&&pass_3j1b&&pass_LeptonCut){
				newtree->Fill();
			}
			if (training&&pt_Lep1>27&&pt_Jet1>40&&pt_Jet2>30&&pt_Jet3>25&&MissingET>30&&\
					massTrans_Lep1MET>40&&pass_3j1b&&pass_LeptonCut&&\
					mass_WlWhBJet<500&&(mass_pseudoWHad>65&&mass_pseudoWHad<92.5)){
				newtree_training->Fill();
			}
		}
		outfile->cd();
		newtree->Write("", TObject::kOverwrite);
		outfile->Close();
		delete outfile;
		if (training) {
			outfile_training->cd();
			newtree_training->Write("", TObject::kOverwrite);
			outfile_training->Close();
		} else {
			system("rm MC_training/reprocessNB_3j1b_"+event+".root");
		}
		delete outfile_training;
	}
	system("echo 'source MergeAndTrim.sh'");
}

