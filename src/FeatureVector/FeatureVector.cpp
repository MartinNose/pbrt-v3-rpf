#include "FeatureVector.h"
#include "geometry.h"
#include "spectrum.h"
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

size_t FeatureVector::width;
size_t FeatureVector::height;
size_t FeatureVector::spp;
float* FeatureVector::randomParameters;
bool FeatureVector::isInitialized;

using namespace pbrt;

FeatureVector::FeatureVector() {
    isInitialized = false;
}
FeatureVector::~FeatureVector() {
    if (isInitialized) {
        delete[] randomParameters;
    }
}

void FeatureVector::initialize(size_t _width, size_t _height, size_t _spp) {
    cv::Mat image;
    image = cv::imread("/Users/liujunliang/Documents/Codes/pbrt-v3-rpf/testres/raw.jpg", 1);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);

    using namespace std;
    width = _width;
    height = _height;
    spp = _spp;
    isInitialized = true;

    randomParameters = new float[width * height * spp * SAMPLELENGTH * sizeof(float)];
    memset(randomParameters, 0, width * height * spp * SAMPLELENGTH * sizeof(float));
    if (randomParameters == NULL) {
        std::cerr << "Failed to allocate memory" << std::endl;
    }
}

float FeatureVector::getPosition(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + offset];
} 
float FeatureVector::getColor(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + offset];
}
float FeatureVector::getFeature(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + offset];
}
float FeatureVector::getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + offset];
}

void FeatureVector::setPosition(size_t x, size_t y, size_t k, float* positions) {
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + WORLD_1_X] = positions[0];
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + WORLD_1_Y] = positions[1];
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + WORLD_1_Z] = positions[2];
}
void FeatureVector::setColor(size_t x, size_t y, size_t k, float* colors) {
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + COLOR_1] = colors[0];
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + COLOR_2] = colors[1];
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + COLOR_3] = colors[2];
}

void FeatureVector::setPosition(size_t x, size_t y, size_t k, Point3f &position) {
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + WORLD_1_X] = position.x;
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + WORLD_1_Y] = position.y;
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + WORLD_1_Z] = position.z;

}
void FeatureVector::setNormal(size_t x, size_t y, size_t k, Normal3f &normal) {
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + NORM_1_X] = normal.x;
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + NORM_1_Y] = normal.y;
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + NORM_1_Z] = normal.z;

}
void FeatureVector::setTexture(size_t x, size_t y, size_t k, Spectrum r) {
    r.ToRGB(randomParameters + (y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + TEXTURE_1_X));
}

void FeatureVector::setColor(size_t x, size_t y, size_t k, Spectrum &R) {
    R.ToRGB(randomParameters + (y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + COLOR_1));
}

// static float setFeature(size_t x, size_t y, size_t k, Scalar* features) {
//     randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + offset] = ;
// }
void FeatureVector::setRandomParameter(size_t x, size_t y, size_t k, OFFSET offset, float para) {
    randomParameters[y * width * spp * SAMPLELENGTH + x * spp * SAMPLELENGTH + k * SAMPLELENGTH + offset] = para;
}

void FeatureVector::write_dat() {
    // Open file for writing
    using namespace std;
    std::ofstream out;
    out.open("sample.dat", std::ios::binary);

    size_t sampleLength = SAMPLELENGTH;
    out.write((char*)(&height),sizeof(size_t));
    out.write((char*)(&width),sizeof(size_t));
    out.write((char*)(&spp),sizeof(size_t));
    out.write((char*)(&sampleLength),sizeof(size_t));

    out.write((char *)(randomParameters), height * width * spp * SAMPLELENGTH * sizeof(float));        
    out.close();
}