#include <iostream>
#include <vector>
#include "mm.hpp"
#include <fstream>
#include <sstream>
using namespace std;
#define inputsize 28*28 //mnist image = 28*28
#define outputsize 10   //0-9

int main(){
	
	vector<int> sh={inputsize,100,outputsize};
	neuralnetwork n1(sh);
	cout << "\nprinting the node bias for each layer\n\n ";
	n1.printlayers();
	cout << "\nprinting the randomized adjacency matrix for the weights\n\n";
	n1.print_adjmat();
	cout << "\n\n";
	return 0;
}
