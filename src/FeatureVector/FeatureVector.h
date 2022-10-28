#pragma once

template<typename Scalar = float>
class FeatureVector {
public:
    void initialize(size_t width, size_t height, size_t spp);

    Scalar getPosition(size_t x, size_t y, size_t k, OFFSET offset);
    Scalar getColor(size_t x, size_t y, size_t k, OFFSET offset);
    Scalar getFeature(size_t x, size_t y, size_t k, OFFSET offset);
    Scalar getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset);

	Scalar setPosition(size_t x, size_t y, size_t k, Scalar position, OFFSET offset);
	Scalar setColor(size_t x, size_t y, size_t k, Scalar color, OFFSET offset);
	Scalar setFeature(size_t x, size_t y, size_t k, Scalar* features, OFFSET offset, size_t size);
	Scalar setFeature(size_t index, Scalar feature);
	Scalar setRandomParameter(size_t x, size_t y, size_t k, Scalar randomParameter, OFFSET offset);

    void setWidth(size_t width);
    void setHeight(size_t height);
    void setSamplesPerPixel(size_t samplesPerPixel);

private:
    FeatureVector();
    ~FeatureVector();

    Scalar* positions;
    Scalar* colors;
    Scalar* features;
    Scalar* randomParameters;

    size_t width;
    size_t height;
    size_t spp;
};