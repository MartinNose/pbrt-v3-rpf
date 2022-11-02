#include "RPF.h"

void initializeFromPbrt(float* pbrtSamples, size_t pbrtWidth, size_t pbrtHeight, size_t pbrtSpp) {
	samples->readSamples(pbrtSamples);
	width = pbrtWidth;
	height = pbrtHeight;
	spp = pbrtSpp;
}

// Algorithm 1: Random Parameter Filtering (RPF) Algorithm
void RPF(CImg<float>* img) {
	boxSizes = new int[4];
	boxSizes[0] = 55;
	boxSizes[1] = 35;
	boxSizes[2] = 17;
	boxSizes[3] = 7;
	for (int t = 0; t < 4; t++) {
		int b = boxSizes[t];
		size_t maxNumOfSamples = (b * b * spp) / 2;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Sample* neighborSamples;
				preprocessSamples(samples, b, maxNumOfSamples, neighborSamples, j, i);
				float* alpha = neighborSamples->getAlpha();
				float* beta = neighborSamples->getBeta();
				float newColor = computeFeatureWeights(t, neighborSamples);
				float finalColor = filterColorSamples(samples, neighborSamples, alpha, beta, newColor);
				samples->updateColor(neighborSamples, j, i, spp);
			}
		}
		samples->setColor();
	}
	boxFilter(img);
}

// Algorithm 2: Preprocess Samples
void preprocessSamples(Sample* samples, int b, size_t maxNumOfSamples, Sample* neighborSamples, int x, int y)

// Algorithm 3: Compute Feature Weights
float computeFeatureWeights(int t, Sample* neighborSamples)

// Algorithm 4: Filter Color Samples
float filterColorSamples(Sample* samples, Sample* neighborSamples, float* alpha, float* beta, float newColor)

void boxFilter(CImg<float>* img) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// filter all samples at a pixel
		}
	}
}