#include "utility.hpp"

imageset::imageset(const string&imagepath, const string&labelpath, bool customnumber, int numimages){
    this->image_file = imagepath;
    this->label_file = labelpath;
    this->num_channels = 1;
    if(customnumber == true){
        this->numimages = numimages;
    }
    load();
}
Tensor3 imageset::get_image(int &index){
    return this->images[index];
}
int imageset::getchannel(){
    return this->num_channels;
}
uint32_t imageset::read_uint32(ifstream &file){
    uint32_t val;
    file.read(reinterpret_cast<char*>(&val), 4);
    return __bswap_32(val);
}
bool imageset::load() {
    
    std::ifstream img(image_file, std::ios::binary);
    std::ifstream lbl(label_file, std::ios::binary);
    if (!img.is_open() || !lbl.is_open()) {
        std::cerr << "Failed to open MNIST files\n";
        return false;
    }
    
    if (read_uint32(img) != 2051 || read_uint32(lbl) != 2049) {
        std::cerr << "Invalid MNIST magic numbers\n";
        return false;
    }

    uint32_t num_images = read_uint32(img);
    uint32_t num_labels = read_uint32(lbl);
    uint32_t rows = read_uint32(img);
    uint32_t cols = read_uint32(img);

    if (num_images != num_labels) {
        std::cerr << "Mismatch in image and label count\n";
        return false;
    }
    if(customnumber==false){
        this->numimages=num_images;
    }

    for (uint32_t n = 0; n < numimages; ++n) {
        Tensor3 image(this->num_channels,Matrix(rows, vector<float>(cols, 0.0f)));
        uint8_t label;
        for(int m= 0; m<num_channels;m++){                                         //number of image channels -> should be 1
            for (uint32_t i = 0; i < rows; ++i) {                                   //number of rows -> MNIST 28
                for (uint32_t j = 0; j < cols; ++j) {                              //number of cols -> MNIST 28
                    uint8_t pixel;
                    img.read(reinterpret_cast<char*>(&pixel), 1);
                    image[m][i][j] = pixel / 255.0f;  // normalize
                }
            }
        }
        this->images.push_back(image);
        lbl.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(label);
       
    }

    return true;
}
Tensor4 imageset::get_imageset(){
    return images;
}
vector<uint8_t> imageset::get_labels(){
    return labels;
}



