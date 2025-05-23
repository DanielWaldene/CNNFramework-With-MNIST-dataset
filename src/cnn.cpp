#include "cnn.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#define epsillon 1e-9
using namespace std;
float MaxP(float &a, float&b){
    return (a>b)?a:b;
}
float dotproducthelper(Matrix &a,Matrix&b){
    int cols_a = (int)a[0].size();
    int rows_a = (int)a.size();
    int rows_b = (int)b.size();
    int cols_b = (int)b[0].size();
    float result= 0.0f;
    for(int i = 0; i< rows_a;i++){
        for(int j = 0; j< cols_a;j++){
           result += a[i][j]*b[i][j];
        }
    }
    return result;
    
}
float cross_entropy_loss(const vector<float>&y_actual, const vector<float>&y_pred){
    float loss = 0.0f;
    for(int i = 0; i<(int)y_actual.size(); i++){
        loss -= y_actual[i] * log(y_pred[i]+epsillon);
    }
    return loss;
}

vector<float> softmax_helper(const vector<float>&logits){
    float max_logit = *max_element(logits.begin(), logits.end());
    vector<float> exps(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (float &v : exps) v /= sum;
    return exps;
}
//this is confusing because i will rerference the sub matrix of the image but 
// in the dot product functions treat it like three pairs of vectors and summ them
float dotproduct(Matrix &a, Matrix&b){

    if(!a.empty()&&!b.empty()&&a[0].size()==b[0].size()&&a.size()==b.size()){
        return dotproducthelper(a,b);
    }
    else{
		throw invalid_argument("Neither a*b or b*a is valid: incompatible dimensions");
	}
}
Matrix autommult(Matrix &a, Matrix &b){
	float reresult;
	int rowsa=(int)a.size();
	int rowsb=(int)b.size();
	int colsa =(int)a[0].size();
	int colsb = (int)b[0].size();
	vector<vector<float>>results(rowsa, vector<float>(colsb, 0.0f));
	for(int i = 0; i< rowsa;i++){
		for(int j = 0;j<colsb ; j++){
			for(int k =0; k < colsa; k++){
				results[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return results;
}
Matrix matmult(Matrix&a, Matrix&b){
	if(!a.empty()&&!b.empty()&&a[0].size()==b.size()){
		cout << "\ndid a*b\n";
		return autommult(a, b);
	}else if(!a.empty() && !b.empty()&& b[0].size()==a.size()){
		cout << "\ndid b*a\n";
		return autommult(b,a);
	}else{
		throw invalid_argument("Neither a*b or b*a is valid: incompatible dimensions");
	}
}
void printTensor3(Tensor3&t){
    for(int i = 0; i< (int)t.size();i++){
        for(int j = 0; j< (int)t[0].size();j++){
            for(int k = 0; k< (int)t[0][0].size();k++){
                cout << setw(6) << setprecision(2)<< t[i][j][k];
            }
            cout <<endl;
        }
        cout <<"\n\n";
    }
}
void printVector(vector<float>&vec){
    for(int i = 0; i< (int)vec.size(); i++){
        cout <<setw(5)<< setprecision(2) <<vec.at(i)<<" ";
    }
    cout << "\n\n";
}
conv::conv(int number_kerns, int out_channels, int kernelsize){
   Tensor4 v(number_kerns, Tensor3(out_channels, Matrix(kernelsize, vector<float>(kernelsize,0.0f))));
    for(int i = 0; i< number_kerns;i++){                 //number of kernels probably 3
        for(int j = 0; j < out_channels; j++){              //number of out_channels-> should be 1
            for(int k = 0; k<kernelsize;k++){               //kernel size -> should be 3
                for(int l = 0; l < kernelsize; l++){
                    float random = static_cast<float>(rand()%100)/100.0f;
                    v[i][j][k][l] = random;
                }
            }
        }
        float bf = static_cast<float>(rand()%100)/100.0f;
        this->bias.push_back(bf);
    }
   this->kernels = v;
}

void conv::printKernels(){
    for(int i =0; i < (int)kernels.size();i++){                         //number of kernels
        for(int j = 0; j< (int)kernels[0].size(); j++){                 //channels -> should be 1
            for(int k =0; k < (int)kernels[0][0].size(); k++){          //kernel width -> should be 3
                for(int l = 0; l< (int)kernels[0][0][0].size();l++){    //kernel height-> should be 3
                    cout << setw(6) << kernels[i][j][k][l]<<" ";        
                }
                cout << endl;
            }
            cout << "\n\n";
        }
    }
}
Tensor3 conv::forward(Tensor3&input){ //pass in one image out of 
    //28x28x1
    //3x3x1 filter -> outputsize= (28 -3)/1+1 assume stride 1 -> 26x2
    this->forward_input = input;
    int kernelsize= (int)this->kernels[0][0].size();
    int num_kernels = (int)this->kernels.size();
    int imagewidth = (int)input[0][0].size();
    int imageheight = imagewidth;
    int outsize = (imagewidth - kernelsize)/1+1;
    Matrix current_image(kernelsize, vector<float>(kernelsize, 0.0f));
    Matrix current_kernel;
    Tensor3 out(num_kernels, Matrix(outsize, vector<float>(outsize, 0.0f)));
    for(int x = 0; x< num_kernels; x++){
        current_kernel = this->kernels[x][0];
        for(int i = 0 ; i < outsize; i++){                  //for the rows in image
            for(int j = 0; j< outsize;j++){                 //for the cols in image
                for(int ki = 0; ki< kernelsize;ki++){
                    for(int kj = 0; kj< kernelsize;kj++){
                       int ip = i+ki;
                       int jp = j+kj;
                       current_image[ki][kj]= input[0][ip][jp];
                    }
                }
                out[x][i][j] = dotproduct(current_image, current_kernel)+this->bias[x];
                
            }
        }
        
    }
    //cout << "input:"<<input.size()<<"x"<< input[0].size()<<"x" <<input[0][0].size()<< "\toutput"<< out.size()<< "x"<<out[0].size()<<"x"<<out[0][0].size();
    return out;
}
Tensor3 conv::backward(Tensor3&dout){
    int num_kernels = this->kernels.size();
    int out_channels = this->kernels[0].size();
    int kernel_size = this->kernels[0][0].size();
    int in_h = forward_input[0].size();
    int in_w = forward_input[0][0].size();
    int out_h = dout[0].size();
    int out_w = dout[0][0].size();
    float lr =0.01f;
    Tensor3 dinput(forward_input.size(),Matrix(in_h, vector<float>(in_w, 0.0f)));
    Tensor4 d_kernels = kernels;
    for(auto& tens3:d_kernels){
        for(auto& mat:tens3){
            for(auto &row:mat){
                fill(row.begin(), row.end(), 0.0f);
            }
        }
    }
    vector<float>d_bias(num_kernels, 0.0f);
    for(int x = 0; x<num_kernels;x++){
        for(int i = 0; i< out_h;i++){
            for(int j = 0; j< out_w;j++){
                for(int ki = 0; ki<kernel_size;ki++){
                    for(int kj=0; kj<kernel_size;kj++){
                        int in_i = i +ki;
                        int in_j = j+kj;
                        dinput[0][in_i][in_j]+=kernels[x][0][ki][kj] * dout[x][i][j];
                        d_kernels[x][0][ki][kj]+=forward_input[0][in_i][in_j]*dout[x][i][j];
                    }
                }
                d_bias[x]+=dout[x][i][j];
            }
        }
    }
    for(int k = 0; k < num_kernels;k++){
        for(int ki = 0; ki<kernel_size;ki++){
            for(int kj= 0; kj<kernel_size;kj++){
                kernels[k][0][ki][kj] -= lr*d_kernels[k][0][ki][kj];
            }
        }
        bias[k] -=lr *d_bias[k];
    }
    return dinput;

}
Tensor3 pooling::forward(Tensor3&input){
    forward_input= input;
    int pooling_size = 2;
    int channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int out_height = input_height / pooling_size;
    int out_width = input_width / pooling_size;

    Tensor3 out(channels, Matrix(out_height, vector<float>(out_width, 0.0f)));
    max_indices.resize(channels, std::vector<std::vector<std::pair<int, int>>>(out_height, std::vector<std::pair<int, int>>(out_width)));
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < out_height; ++i) {
            for (int j = 0; j < out_width; ++j) {
                float max_val = input[c][i * pooling_size][j * pooling_size];
                int max_ki = 0;
                int max_kj = 0;
                for (int ki = 0; ki < pooling_size; ++ki) {
                    for (int kj = 0; kj < pooling_size; ++kj) {
                        int row = i * pooling_size + ki;
                        int col = j * pooling_size + kj;
                        if (input[c][row][col] > max_val) {
                            max_val = input[c][row][col];
                            max_ki = ki;
                            max_kj = kj;
                        }
                    }
                }
                out[c][i][j] = max_val;
                max_indices[c][i][j] = std::make_pair(i *pooling_size+max_ki, j*pooling_size+max_kj);
            }
        }
    }
    return out;
}
Tensor3 pooling::backward(Tensor3&d_out){
    int pooling_size = 2;
    int channels = forward_input.size();
    int in_h = forward_input[0].size();
    int in_w = forward_input[0][0].size();
    int out_h = d_out[0].size();
    int out_w = d_out[0][0].size();
    Tensor3 d_input(channels, Matrix(in_h, vector<float>(in_w, 0.0f)));
    for(int c = 0; c<channels;c++){
        for(int i = 0; i < out_h;i++){
            for(int j = 0; j < out_w;j++){
                auto [max_i, max_j] = max_indices[c][i][j];
                d_input[c][max_i][max_j] = d_out[c][i][j];
            }
        }
    }
    return d_input;
}
vector<float> flatten::forward(Tensor3&input){ 
    if(input.empty())cout << "\nflatten input is empty\n";
    last_shape = {(int)input.size(), (int)input[0].size(), (int)input[0][0].size()};
    vector<float>flat;
    for(auto channel: input){
        for(auto row: channel){
            for(auto col:row){
                flat.push_back(col);
            }
        }
    }
    //cout << "input:"<<input.size()<< "x"<<input[0].size()<<"x" <<input[0][0].size()<< "\toutput"<< flat.size()<<"x"<<1;
    return flat;
}
Tensor3 flatten::backward(vector<float>&grad){
    Tensor3 din(this->last_shape[0], Matrix(this->last_shape[1], vector<float>(this->last_shape[2], 0.0f)));
    int idx=0;
    for(int c = 0 ; c< last_shape[0];c++){
        for(int i = 0; i< last_shape[1]; i++){
            for(int j = 0; j< last_shape[2]; j++){
                din[c][i][j] = grad[idx++];
            }
        }
    }
    return din;
}
Tensor3 relu::forward(Tensor3&input){ 
    if(input.empty())cout <<"\n relu forward input is empty\n ";
    this->forward_input = input;
    for(auto& channel:input){
        for(auto& row:channel){
            for(auto& col:row){
                col = max(0.0f, col);
            }
        }
    }
    // cout << "input:"<<input.size()<<"x" <<input[0].size()<<"x"<< input[0][0].size();
    return input;
}
Tensor3 relu::backward(Tensor3&dout){
    Tensor3 din = dout;
    for (int c = 0; c < (int)forward_input.size(); c++) {
        for (int i = 0; i < (int)forward_input[0].size(); i++) {
            for (int j = 0; j < (int)forward_input[0][0].size(); j++) {
                if (forward_input[c][i][j] <= 0){
                    din[c][i][j] = 0;
                }
            }
        }
    }
    return din;
    
}
vector<float> softmax::forward(vector<float>&input){ 
    vector<float> out(input.size(), 0.0f);
    if(input.empty()){
        cout << "\n soft max empty \n";
        return{};
    }
    float max=input[0];
    float sum =0.0f;
    for(int i = 0; i< (int)input.size();i++){
        if(input[i]>max)max = input[i];
    }
    for(int i = 0;i<(int)input.size();i++){
        out[i] = exp(input[i]-max);
        sum+=out[i];
    }
    for(int i = 0;i<(int)input.size();i++){
        out[i] = out[i] / sum;
    }
    this->o = out;
    return out;
}
vector<float> softmax::backward(vector<float>&input){ //loss +crossentropy is done in training loop so just pass to next layer
    return input;
}
dense::dense(int output_size){
    bias = vector<float> (output_size, 0.0f);
    init = false;
}
vector<float> dense::forward(vector<float>&input){
    last_input = input;
    if(!init){
        input_size = (int)input.size();
        weights =Matrix((int)bias.size(),vector<float>(input_size, 0.0f));
        for(int i = 0; i< (int)bias.size();i++){
            for(int j = 0; j< input_size;j++){
                weights[i][j] = static_cast<float>(rand()%100)/100.0f;
            }
        }
        init = true;
    }
    vector<float> output(weights.size(), 0.0f);
    for(int i = 0; i<(int)weights.size();i++){
        for(int j = 0; j<(int)input.size();j++){
            output[i]+=weights[i][j]*input[j];
        }
        output[i]+=bias[i];
    }
    // cout << "input:"<<input.size()<<"x1"<< "\toutput"<< output.size()<< "x1";
    return output;
}
vector<float> dense::backward(vector<float>&dout){  //input actual label as float
    float lr = 0.01f; // learning rate
    vector<float> din(input_size, 0.0f); // gradient to pass to previous layer

    // Update weights and biases, and compute gradient for previous layer
    for (int i = 0; i < (int)weights.size(); i++) {
        for (int j = 0; j < input_size; j++) {
            din[j] += weights[i][j] * dout[i]; // dL/dx = W^T * dL/dy
            weights[i][j] -= lr * dout[i] * last_input[j]; // SGD update
        }
        bias[i] -= lr * dout[i];
    }
    return din;
}

