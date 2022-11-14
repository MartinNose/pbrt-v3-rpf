#pragma once

#ifndef _SRC_FEATUREVECTOR_
#define _SRC_FEATUREVECTOR_

#include <iostream>
#include <cstdlib>
#include <string>
#include "pbrt.h"
#include "geometry.h"
#include "global.h"

using namespace pbrt;
class FeatureVector {
private:
    // static float* positions;
    // static float* colors;
    // static float* features;
    static float* randomParameters;

    static size_t width;
    static size_t height;
    static size_t spp;
    static bool isInitialized;
public:
    FeatureVector();
    ~FeatureVector();

    static void initialize(size_t _width, size_t _height, size_t _spp);
    static float getPosition(size_t x, size_t y, size_t k, OFFSET offset);
    static float getColor(size_t x, size_t y, size_t k, OFFSET offset);
    static float getFeature(size_t x, size_t y, size_t k, OFFSET offset);
    static float getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset);

	static void setPosition(size_t x, size_t y, size_t k, float* positions);
	static void setPosition(size_t x, size_t y, size_t k, Point3f &position);
	static void setNormal(size_t x, size_t y, size_t k, Normal3f &normal);
	static void setColor(size_t x, size_t y, size_t k, Spectrum &R);
	static void setColor(size_t x, size_t y, size_t k, float* colors);
    static void setTexture(size_t x, size_t y, size_t k, Spectrum r);
	static void setRandomParameter(size_t x, size_t y, size_t k, OFFSET offset, float para);

    static void write_dat();
};

#endif