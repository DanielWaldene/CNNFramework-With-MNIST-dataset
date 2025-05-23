#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cstdint>
#include <optional>
#include <cassert>
using namespace std;
using Matrix = vector<vector<float>>;
using Tensor3 = vector<vector<vector<float>>>;
using Tensor4 = vector<vector<vector<vector<float>>>>;
const string trainingfile = "data/train-images.idx3-ubyte";
const string traininglabels = "data/train-labels.idx1-ubyte";
const string testingfile = "data/t10k-images.idx3-ubyte";
const string testinglabels = "data/t10k-labels.idx1-ubyte";
class imageset{
    public:
        imageset(const string&imagepath, const string&labelpath, bool customnumber ,int num_images);
        Tensor4 get_imageset();
        vector<uint8_t> get_labels();
        Tensor3 get_image(int &index);
        int getchannel();
    private:
        bool load();
        uint32_t read_uint32(ifstream &file);
        string image_file;
        string label_file;
        int num_channels;
        Tensor4 images;
        vector<uint8_t> labels;
        bool customnumber;
        int numimages;
};

