#include "FeatureVector.h"
#include "geometry.h"
#include "spectrum.h"

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
        delete randomParameters;
    }
}

void FeatureVector::initialize(size_t _width, size_t _height, size_t _spp) {
    width = _width;
    height = _height;
    spp = _spp;
    isInitialized = true;


    randomParameters = new float[width * height * spp * 28 * sizeof(float)];
    if (randomParameters == NULL) {
        std::cerr << "Failed to allocate memory" << std::endl;
    }
}

float FeatureVector::getPosition(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
} 
float FeatureVector::getColor(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
}
float FeatureVector::getFeature(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
}
float FeatureVector::getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset) {
    return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
}

void FeatureVector::setPosition(size_t x, size_t y, size_t k, float* positions) {
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_X] = positions[0];
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_Y] = positions[1];
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_Z] = positions[2];
}
void FeatureVector::setColor(size_t x, size_t y, size_t k, float* colors) {
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + COLOR_1] = colors[0];
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + COLOR_2] = colors[1];
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + COLOR_3] = colors[2];
}

void FeatureVector::setPosition(size_t x, size_t y, size_t k, Point3f &position) {
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_X] = position.x;
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_Y] = position.y;
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_Z] = position.z;

}
void FeatureVector::setNormal(size_t x, size_t y, size_t k, Point3f &normal) {
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + NORM_1_X] = normal.x;
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + NORM_1_Y] = normal.y;
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + NORM_1_Z] = normal.z;

}
void FeatureVector::setTexture(size_t x, size_t y, size_t k, Spectrum r) {
    r.ToRGB(randomParameters + (y * width * spp* 28 + x * spp * 28 + k * 28 + TEXTURE_1_X));
}

// static float setFeature(size_t x, size_t y, size_t k, Scalar* features) {
//     randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset] = ;
// }
void FeatureVector::setRandomParameter(size_t x, size_t y, size_t k, OFFSET offset, float para) {
    randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset] = para;
}

void FeatureVector::write_dat() {
    
}