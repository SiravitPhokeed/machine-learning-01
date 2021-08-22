#include <iostream>
#include <vector>
using namespace std;

#include "MLP.h"

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

int main(int argc, char **argv) {
	#include "TrainingData.cpp"
	MLP mlp(4, 2, 2, 0.5);
	mlp.LoadWeight("beans.txt");
	mlp.SaveWeight("beans.txt");
	
	// for (int i = 0; i < numTrainingSets; i++) {
	// 	// for (int j = 0; j < numInputs; j++)
	// 	// 	cout << training_inputs[i][j] << endl;
	// 	mlp.Training(training_inputs[i], training_outputs[i]);
	// }

	// if (mlp.Training(training_inputs[0], training_outputs[0]))

	return 0;
}