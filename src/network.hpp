#pragma once
#include "cnn.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <random>
#include <numeric>
class network{
    public:
        void addlayer(layer* v);
        void train(Tensor4 &images, vector<uint8_t>&labels);
        float inference(Tensor4 &images, vector<uint8_t>&labels);
    private:
        vector<float> forward(Tensor3& input);
        void backward(vector<float> &input);
        vector<layer*>structure;
        
};