#include "network.hpp"

int findmaxelement(vector<float>&pred){
	float max = 0.0;
	int max_index=0;
	for(int i = 0; i<(int)pred.size(); i++){
		if(pred.at(i) > max){
			max= pred.at(i);
			max_index = i;
		}
	}
	return max_index;
}
void network::addlayer(layer *v){
    this->structure.push_back(v);
}

void network::train(Tensor4 &images, vector<uint8_t>&labels){
	for(int i = 0; i< (int)images.size()-1;i++){
		
			//cout << "EPOCH " << i << "in progress";
			vector<float> prediction = forward(images.at(i));
			vector<float> target(prediction.size(), 0.0f);
			target[labels[i]] = 1.0f;
			float loss = 0.0f;
			for (int j = 0; j < (int)prediction.size(); ++j){
				loss -= target[j]* log(prediction[j])+1e-9;
			}
			vector<float>grad = prediction;
			grad[labels[i]] -=1.0f;

			backward(grad);
		
	}

}
float network::inference(Tensor4&images, vector<uint8_t>&labels){
	vector<int> indices(images.size());
    iota(indices.begin(), indices.end(), 0);
	random_device rd;
	mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

	int class_;
	float num_correct=0.0;
	float num_incorrect= 0.0;
	for(int i = 0;i < (int)images.size();i++){
		vector<float> prediction = forward(images.at(i));
		class_ = findmaxelement(prediction);
		int actual = static_cast<int>(labels.at(i));
		if(class_ == actual){
			num_correct++;
		}
		else{
			num_incorrect++;
		}
	}
	return num_correct/(num_incorrect+num_correct);
}
vector<float> network::forward(Tensor3 &input){
    vector<float> vinput;
	for(auto l : this->structure){
        if(dynamic_cast<conv*>(l)){
			//cout << "conv layer"<< endl;
			input = dynamic_cast<conv*>(l)->forward(input);
			//printTensor3(input);
		}else if(dynamic_cast<pooling*>(l)){
			//cout << "pool layer"<< endl;
			input = dynamic_cast<pooling*>(l)->forward(input);
			//printTensor3(input);
		}else if(dynamic_cast<relu*>(l)){
			//cout << "relu layer"<<endl;
			input = dynamic_cast<relu*>(l)->forward(input);
			//printTensor3(input);
		}
			// outputs change type
		else if(dynamic_cast<flatten*>(l)){
			//cout<< "flatten layer"<<endl;
			vinput = dynamic_cast<flatten*>(l)->forward(input);
			//printVector(vinput);
		}
		else if(dynamic_cast<softmax*>(l)){
			//cout << "softmax layer"<< endl;
			vinput = dynamic_cast<softmax*>(l)->forward(vinput);
			//printVector(vinput);
		}
		else if(dynamic_cast<dense*>(l)){
			//cout << "dense layer" << endl;
			vinput = dynamic_cast<dense*>(l)->forward(vinput);
			//printVector(vinput);
		}
	}
	return vinput;
}
void network::backward(vector<float>&grad){
	Tensor3 tgrad;
	vector<float> vgrad = grad;
	for(int i = (int)structure.size()-1; i>=0; i--){
		if(dynamic_cast<conv*>(this->structure.at(i))){
			//cout<< "conv layer(back)"<<endl; 	
			tgrad =dynamic_cast<conv*>(this->structure.at(i))->backward(tgrad);
			//printTensor3(tgrad);
		}
		else if(dynamic_cast<pooling*>(this->structure.at(i))){
			//cout<<"pool layer(back)"<<endl;
			tgrad = dynamic_cast<pooling*>(this->structure.at(i))->backward(tgrad);
			//printTensor3(tgrad);
		}
		else if(dynamic_cast<relu*>(this->structure.at(i))){
			//cout<<"relu layer(back)"<<endl;
			tgrad = dynamic_cast<relu*>(this->structure.at(i))->backward(tgrad);
			//printTensor3(tgrad);
		}
		else if(dynamic_cast<flatten*>(this->structure.at(i))){
			//cout<<"flatten layer(back)"<<endl;
			tgrad=dynamic_cast<flatten*>(this->structure.at(i))->backward(vgrad);
			//printTensor3(tgrad);
		}
		else if(dynamic_cast<softmax*>(this->structure.at(i))){
			//cout<<"softmax layer(back)"<<endl;
			vgrad = dynamic_cast<softmax*>(this->structure.at(i))->backward(vgrad);
			//printVector(vgrad);
		}
		else if(dynamic_cast<dense*>(this->structure.at(i))){
			//cout<<"dense layer(back)"<<endl;
			vgrad = dynamic_cast<dense*>(this->structure.at(i))->backward(vgrad);
			//printVector(vgrad);
		}
	}
}

