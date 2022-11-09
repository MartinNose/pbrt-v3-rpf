#pragma once

#include <iostream>
#include <cstdlib>
#include "global.h"

template<typename Scalar = float>
class FeatureVector {
public:
    void initialize(size_t _width, size_t _height, size_t _spp) {
        width = _width;
        height = _height;
        spp = _spp;
        isInitialized = true;


        randomParameters = new Scalar[width * height * spp * 28 * sizeof(Scalar)];
        if (randomParameters == NULL) {
            std::cerr << "Failed to allocate memory" << std::endl;
        }
    }

    static Scalar getPosition(size_t x, size_t y, size_t k, OFFSET offset) {
        return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
    } 
    static Scalar getColor(size_t x, size_t y, size_t k, OFFSET offset) {
        return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
    }
    static Scalar getFeature(size_t x, size_t y, size_t k, OFFSET offset) {
        return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
    }
    static Scalar getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset) {
        return randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset];
    }

	static Scalar setPosition(size_t x, size_t y, size_t k, Scalar* positions) {
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_X] = positions[0];
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_Y] = positions[1];
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + WORLD_1_Z] = positions[2];
    }
	static Scalar setColor(size_t x, size_t y, size_t k, Scalar* colors) {
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + COLOR_1] = colors[0];
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + COLOR_2] = colors[1];
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + COLOR_3] = colors[2];
    }
	// static Scalar setFeature(size_t x, size_t y, size_t k, Scalar* features) {
    //     randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset] = ;
    // }
	static Scalar setRandomParameter(size_t x, size_t y, size_t k, OFFSET offset, Scalar para) {
        randomParameters[y * width * spp* 28 + x * spp * 28 + k * 28 + offset] = para;
    }

    // static void setWidth(size_t width);
    // static void setHeight(size_t height);
    // static void setSamplesPerPixel(size_t samplesPerPixel);

private:
    FeatureVector() {
        isInitialized = false;
    }
    ~FeatureVector() {
        if (isInitialized) {
            delete randomParameters;
        }
    }

    // static Scalar* positions;
    // static Scalar* colors;
    // static Scalar* features;
    static Scalar* randomParameters;

    static size_t width;
    static size_t height;
    static size_t spp;
    bool isInitialized;
};