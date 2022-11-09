#include "RPF.h"

void initializeFromPbrt(FeatureVector* pbrtSamples, size_t pbrtWidth,
                        size_t pbrtHeight, size_t pbrtSpp, int numFeatures) {
	// Define in header
	samples = pbrtSamples;
	width = pbrtWidth;
	height = pbrtHeight;
	spp = pbrtSpp;
    mFeatures = numFeatures;
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
				FeatureVector* neighborSamples;
				preprocessSamples(samples, b, maxNumOfSamples, neighborSamples, j, i);
				float* alpha = neighborSamples->getAlpha();
				float* beta = neighborSamples->getBeta();
				float newColor = computeFeatureWeights(t, neighborSamples);
				float finalColor = filterColorSamples(samples, neighborSamples, alpha, beta, newColor);
                samples->setColor(j, i, 0, finalColor, spp);
			}
		}
		samples->setColor();
	}
	boxFilter(img);
}

// Algorithm 2: Preprocess Samples
void preprocessSamples(FeatureVector* samples, int b, size_t maxNumOfSamples,
	FeatureVector* neighborSamples, int x, int y) {
	float sigmaP = b / 4.0f;
    neighborSamples = samples;

	// Compute mean (mfP) and standard deviation (σfP) of the features of samples in pixel P for clustering

	// Add samples to neighborhood
    for (int q = 0; q < maxNumOfSamples - spp; q++) {
		// Select a random sample j from samples inside the box but outside P with distribution based on σp
        flag = true;
        // Perform clustering
        for (int k = 0; k < mFeatures; k++) {
			
		}
    }
	// Compute normalized vector for each sample by removing mean and dividing by standard deviation
    neighborSamples->getStatistics();
}

// Algorithm 3: Compute Feature Weights
    float computeFeatureWeights(int t, FeatureVector* neighborSamples)

// Algorithm 4: Filter Color Samples
    float filterColorSamples(FeatureVector* samples,
                             FeatureVector* neighborSamples,
                             float* alpha, float* beta, float newColor)

void boxFilter(CImg<float>* img) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float c[3] = {0.0f, 0.0f, 0.0f};
            for (size_t k = 0; k < samplesPerPixel; k++) {
				c[0] += samples->getColor(i, j, k, 0);
                c[1] += samples->getColor(i, j, k, 1);
                c[2] += samples->getColor(i, j, k, 2);
            }
            for (int k = 0; k < 3; k++) {
                (*img)(j, i, 0, k) = color[k] / spp;
			}
		}
	}
}