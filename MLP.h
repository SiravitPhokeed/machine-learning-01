#ifndef MLP_H
#define MLP_H

#include <math.h>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#define LearningRate 0.1

using namespace std;

class MLPCell {
		double delta;
		double bias;			
        double sigmoid(double x) { return 1/(1 + exp(-x)); }
        double dSigmoid(double x) { return x*(1-x); }
	public:
		vector<double> input;
		vector<double> inerr;
		vector<double> weight;
		double output;
		MLPCell(int inputNum);
		void FeedForward();
		void BackPropagate(double derr);			
		void AdjustWeight(double lr);		
			
};

class MLP {
		vector<MLPCell> hiddenLayer;
		vector<MLPCell> outputLayer;
		double myThreshold;
		double Step(double value) {if(value<myThreshold) return 0.0; else return 1.0;}
	public:
		vector<double> input;
		vector<double> output;
		MLP(int inputNum,int hiddenNum,int outputNum,double threshold);
		void Testing();
		bool Training(double trainingInput[], double trainingOutput[]);	
		void SaveWeight(string FileName);                                 //Assignment1
		void LoadWeight(string FileName);                                 //Assignment2    	
};

#endif

MLPCell::MLPCell(int inputNum){
	srand((unsigned)time(NULL));
	bias = 0;
	input.resize(inputNum);
	inerr.resize(inputNum);
	for (int i=0;i<inputNum;i++)
		weight.push_back(rand()/RAND_MAX);		
}


void MLPCell::FeedForward(){
	double sum = bias;
	for (int i=0;i<input.size();i++)
			sum += input[i]*weight[i];
	output = sigmoid(sum);
}

void MLPCell::BackPropagate(double derr){
	delta = derr*dSigmoid(output);
	for (int i=0;i<inerr.size();i++)
		inerr[i] = delta * weight[i];
}

void MLPCell::AdjustWeight(double lr){
	bias += delta*lr;
	for (int i=0;i<weight.size();i++)
		weight[i] += input[i]*delta*lr;
}

//=================== MLP ========================

MLP::MLP(int inputNum,int hiddenNum,int outputNum,double threshold){
	hiddenLayer.resize(hiddenNum,MLPCell(inputNum));
	outputLayer.resize(outputNum,MLPCell(hiddenNum));
	input.resize(inputNum);
	output.resize(outputNum);
	myThreshold = threshold;
}


bool MLP::Training(double trainingInput[], double trainingOutput[]){
	// this part below doesn't work :(
	// sizeof an array with 4 things returns 8 (should be 32)
	
	// if (sizeof(trainingInput)/sizeof(double) != input.size() || 
	//     sizeof(trainingOutput)/sizeof(double) != output.size()) {
	//     	cout << "Training data range not match!!" << endl;
	//     	return false;
	// } 

	for (int i=0;i<input.size();i++)
		input[i]=trainingInput[i];	
	do{
		Testing();
		double sumerr=0;		
		for (int i=0;i<outputLayer.size();i++)	
			sumerr+=abs(trainingOutput[i]-output[i]);
		if(sumerr==0) break;
		for(int i=0;i<outputLayer.size();i++){
			outputLayer[i].BackPropagate(trainingOutput[i]-outputLayer[i].output);
			outputLayer[i].AdjustWeight(0.01);			
		}
		for(int i=0;i<hiddenLayer.size();i++){
			double sumInerr=0;
			for(int j=0;j<outputLayer.size();j++)
				sumInerr+=outputLayer[j].inerr[i];
			hiddenLayer[i].BackPropagate(sumInerr);	
			hiddenLayer[i].AdjustWeight(0.01);
		}	
	}while(true);		
	return true;	
}

void MLP::Testing(){	
	// -------------- testing hedden layer -------------------
	for (int i=0;i<hiddenLayer.size();i++){
		for(int j=0;j<input.size();j++) 
			hiddenLayer[i].input[j]=input[j];	
		hiddenLayer[i].FeedForward();	
	}
	//--------------- testing output layer -------------------
	for (int i=0;i<outputLayer.size();i++){
		for(int j=0;j<hiddenLayer.size();j++) 
			outputLayer[i].input[j]=hiddenLayer[j].output;	
		outputLayer[i].FeedForward();	
		output[i] = Step(outputLayer[i].output);
	}
}

void MLP::SaveWeight(string FileName) {
	cout << "Saving to " << FileName << endl << endl;

	fstream SaveFile;
	SaveFile.open(FileName, fstream::out);

	SaveFile << "MLP Weights 🍌" << endl;

	SaveFile << "Init:" << endl;
	SaveFile
		<< input.size()       << " " 
		<< hiddenLayer.size() << " "
		<< outputLayer.size() << " "
		<< myThreshold        << " " << endl;

	SaveFile << "Weights:" << endl;

	cout << "Input->hidden weights: {" << endl;
	for (int i=0; i<hiddenLayer.size(); i++) {
		cout << "  { ";
		for (int j=0; j<input.size(); j++) {
			SaveFile << hiddenLayer[i].weight[j] << " ";
			cout << hiddenLayer[i].weight[j] << ", ";
		}
		SaveFile << endl;
		cout << "}," << endl;
	}
	cout << "}" << endl << endl;

	cout << "Hidden->output weights: {" << endl;
	for (int i=0; i<outputLayer.size(); i++) {
		cout << "  { ";
		for(int j=0; j<hiddenLayer.size(); j++) {
			SaveFile << outputLayer[i].weight[j] << " ";
			cout << hiddenLayer[i].weight[j] << ", ";
		}
		SaveFile << endl;
		cout << "}," << endl;
	}
	cout << "}" << endl;

	SaveFile.close();
}

void MLP::LoadWeight(string FileName) {
	cout << "Loading from " << FileName << endl;
	int inputNum;
	int hiddenNum;
	int outputNum;
	double threshold;
	string currLine;
	fstream LoadFile;

	LoadFile.open(FileName, fstream::in);
	
	getline(LoadFile, currLine);
	if (currLine.compare("MLP Weights ≡ƒìî")) {  // C++ read "🍌" as "≡ƒìî" for some reason
		cout << "Signature check passed." << endl << endl;

		getline(LoadFile, currLine);
		if (currLine.compare("Init: ")) {
			cout << "Loading init..." << endl;
			for (int i=0; i<4; i++) {
				getline(LoadFile, currLine, ' ');
				switch (i) {
					case 0: inputNum  = stoi(currLine); break;
					case 1: hiddenNum = stoi(currLine); break;
					case 2: outputNum = stoi(currLine); break;
					case 3: threshold = stof(currLine); break;
				}
			}

			cout << "No. of inputs:        " << inputNum  << endl;
			cout << "No. of hidden cells:  " << hiddenNum << endl;
			cout << "No. of outputs:       " << outputNum << endl;
			cout << "Threshold:            " << threshold << endl << endl;

			getline(LoadFile, currLine);

			if (currLine.compare("Weights: ")) {
				cout << "Loading weights..." << endl;
				getline(LoadFile, currLine);

				for (int i=0; i<hiddenNum; i++) {
					for (int j=0; j<inputNum; j++) {
						getline(LoadFile, currLine, ' ');
						hiddenLayer[i].weight[j] = stod(currLine);
					}
				}
				for (int i=0; i<outputNum; i++) {
					for(int j=0; j<hiddenNum; j++) {
						getline(LoadFile, currLine, ' ');
						outputLayer[i].weight[j] = stod(currLine);
					}
				}

				cout << "Weights loaded successfully." << endl << endl;

			} else {
				cout << "Weights load failed." << endl;
			}

		} else {
			cout << "Init load failed." << endl;
		}
	} else {
		cout << "Invalid file!" << endl;
	}
}