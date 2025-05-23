#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
using namespace std;
using Matrix = vector<vector<float>>;
using Tensor3 = vector<vector<vector<float>>>;
using Tensor4 = vector<vector<vector<vector<float>>>>;


float dotproduct(Matrix &a,Matrix&b);
float dotproducthelper(Matrix&a,Matrix&b);
Matrix autommult(Matrix &a, Matrix &b);
Matrix matmult(Matrix&a, Matrix&b);
void printTensor3(Tensor3&t);
void printVector(vector<float>&vec);
class layer{
    public:
        virtual Tensor3 forward(const Tensor3&input){
            cout << "forward from base layer class";
        }
        virtual Tensor3 backward(const Tensor3&input){
            cout << "backward from base layer class";
        }
    private:
        Tensor4 weights;
        vector<float>bias;
};

class conv:public layer{
    public:
        conv(int number_kernels =3, int outchannels=1, int kernelsize=3);
        Tensor3 forward(Tensor3&input);
        Tensor3 backward(Tensor3&input);
        void printKernels();
        
    private: 
        Tensor3 forward_input;
        Tensor4 kernels;
        vector<float> bias;
        
};
class pooling:public layer{
    public:
        Tensor3 forward(Tensor3&input);
        Tensor3 backward(Tensor3&input);
    private:
        Tensor3 forward_input;
        std::vector<std::vector<std::vector<std::pair<int, int>>>> max_indices;
        
};
class flatten:public layer{
    public:
        vector<float> forward(Tensor3&input);
        Tensor3 backward(vector<float>&input);
    private:
        vector<int> last_shape;
};
class dense:public layer{
    public:
        dense(int outputsize=10);
        vector<float> forward(vector<float>&input);
        vector<float> backward(vector<float>&input);
    private:
        bool init;
        int input_size; //alot of change to the structure of the network will cause input size to change so better to do it in runtime;
        Matrix weights;
        vector<float> bias;
        vector<float> last_input;
};
class relu:public layer{
    public:
        Tensor3 forward(Tensor3&input);
        Tensor3 backward(Tensor3&input);
    private:
        Tensor3 forward_input;
};
class softmax:public layer{
    public:
        vector<float> forward(vector<float>&input);
        vector<float> backward(vector<float>&input); //compute loss in training function and pass to this
    private:
        vector<float> o;
};
