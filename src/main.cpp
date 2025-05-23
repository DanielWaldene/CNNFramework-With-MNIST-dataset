#include <iostream>
#include <vector>
#include "eigencnn.hpp"
#include "cnn.hpp"
#include "utility.hpp"
#include "network.hpp"
#include <fstream>
#include <sstream>
using namespace std;

void imageset_testing(bool cust, int num){
	imageset data(trainingfile, traininglabels, cust, num);
	Tensor4 localdata;
	localdata =data.get_imageset();
	int numchannels = data.getchannel();
	cout << " the number of images = "<<localdata.size()<<endl;
	cout << " the number of channels = "<<localdata[0].size()<<endl;
	cout << "the number of rows = " <<localdata[0][0].size()<<endl;
	cout << "the number of columns = " << localdata[0][0][0].size()<<endl;
	for(int i = 0; i< num; i++){
		for(int j = 0; j < (int)localdata[0].size(); j++){							//should be 1
			for(int k = 0; k< (int)localdata[0][0].size();k++){					//should be 28
				for(int l = 0; l< (int)localdata[0][0][0].size();l++){				//should be 28
					cout << setw(6)<< setprecision(2) <<localdata[i][j][k][l]<<" ";
				}
				cout << endl;
			}
			cout << endl;
		}
		cout <<"\n\n";
	}
}

void conv_testing(int number_kerns, int out_chann, int kern_size){
	conv*c = new conv(number_kerns,  out_chann,  kern_size);
	c->printKernels();
}
void conv_forward_testing(bool cust, int num, int numkerns, int outchann, int kernsize){
	imageset data(trainingfile, traininglabels, cust, num);
	Tensor4 localdata = data.get_imageset();
	Tensor3 o;
	layer*c = new conv(numkerns, outchann, kernsize);
	for(int i = 0; i< (int)localdata.size();i++){
		for(int j = 0; j < (int)localdata[0].size();j++){
			o = c->forward(localdata[i]);
			printTensor3(o);
		}
	}

}
int main(){
	bool cust = true;
	bool cust_test = true;
	int num_test = 10000;
	int num = 50000;
	imageset testdata(testingfile, testinglabels, cust_test, num_test);
	Tensor4 localTest = testdata.get_imageset();
	vector<uint8_t>testlabels = testdata.get_labels();

	imageset data(trainingfile, traininglabels, cust, num);
	Tensor4 localdata = data.get_imageset();
	vector<uint8_t>labels = data.get_labels();
	network net;
	net.addlayer(new conv());
	net.addlayer(new relu());
	net.addlayer(new pooling());
	net.addlayer(new flatten());
	net.addlayer(new dense());
	net.addlayer(new softmax());
	net.train(localdata, labels);
	float percent_correct = net.inference(localTest, testlabels);
	cout << "\npercent correct with inference: "<< percent_correct *100<< endl;
	
	
	return 0;
}