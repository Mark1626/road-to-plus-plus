// Reference

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>

/*
Haar Wavelet

1

0.5           ┏━━━━━━━━━━━━━━━━━━
              ┃
              ┃
0   ━━━━━━━━━━┛
*/
class HaarWavelet {
    static const int waveletsize = 3;
    static const int size = 6;

    double wavelet[waveletsize] = {0.0, 1.0/2.0, 1.0/2.0};
    double sigmafactors[size + 1] = {
      1.00000000000, 7.07167810e-1, 5.00000000e-1, 3.53553391e-1,
      2.50000000e-1, 1.76776695e-1, 1.25000000e-1
    };

    public:
    HaarWavelet() {}

    int getNumScales(size_t length) {
        return 1 + int(log(double(length - 1) / double(size - 1)) / M_LN2);
    }

    int getMaxSize(int scale) {
        return int(pow(2, scale - 1)) * (size - 1) + 1;
    }

    int maxFactor() {
        return size;
    }

    double sigmaFactor(int scale) {
        return sigmafactors[scale];
    }

    int width() {
        return size;
    }

    double coeff(int i) {
        return wavelet[i];
    }
};

struct Param {
    public:
    float reconSNR;
    int minScale;
    int maxScale;
    Param(float reconSNR) : reconSNR(reconSNR) {}
};

#pragma omp declare target
inline void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

float qselect(float *arr, int len, int nth) {
  int start = 0;
  for (int index = 0; index < len - 1; index++) {
    if (arr[index] > arr[len - 1])
      continue;
    swap(arr[index], arr[start]);
    start++;
  }
  swap(arr[len - 1], arr[start]);

  if (nth == start)
    return arr[start];

  return start > nth ? qselect(arr, start, nth)
                     : qselect(arr + start, len - start, nth - start);
}
#pragma omp end declare target

#pragma omp declare target
float findMedian(float* input, size_t xdim) {
    float median;

    float* arr = (float*) malloc(sizeof(float) * xdim);
    for (int i = 0; i < xdim; i++)
        arr[i] = input[i];

    median = qselect(arr, xdim, xdim/2);
    if (xdim % 2 == 0) {
        median += qselect(arr, xdim, xdim/2 - 1);
        median /= 2;
    }
    free(arr);
    return median;
}
#pragma omp end declare target

#pragma omp declare target
float findMadfm(float* input, size_t xdim) {
    float median = findMedian(input, xdim);

    float* arr = (float*) malloc(sizeof(float) * xdim);
    for (int i = 0; i < xdim; i++) {
        float val = input[i] - median;
        arr[i] = val < 0 ? -val : val;
    }

    median = qselect(arr, xdim, xdim/2);
    if (xdim % 2 == 0) {
        median += qselect(arr, xdim, xdim/2 - 1);
        median /= 2;
    }
    free(arr);
    return median;
}
#pragma omp end declare target

#pragma omp declare target
float findMedianDiff(float* first, float *second, size_t xdim) {
    float median;

    float* arr = (float*) malloc(sizeof(float) * xdim);
    for (int i = 0; i < xdim; i++)
        arr[i] = first[i] - second[i];

    median = qselect(arr, xdim, xdim/2);
    if (xdim % 2 == 0) {
        median += qselect(arr, xdim, xdim/2 - 1);
        median /= 2;
    }
    free(arr);
    return median;
}
#pragma omp end declare target

#pragma omp declare target
float findMadfmDiff(float* first, float* second, size_t xdim) {
    float median = findMedianDiff(first, second, xdim);

    float* arr = (float*) malloc(sizeof(float) * xdim);
    for (int i = 0; i < xdim; i++) {
        float val = first[i] - second[i] - median;
        arr[i] = val < 0 ? -val : val;
    }

    median = qselect(arr, xdim, xdim/2);
    if (xdim % 2 == 0) {
        median += qselect(arr, xdim, xdim/2 - 1);
        median /= 2;
    }
    free(arr);
    return median;
}
#pragma omp end declare target

// This should be given to cores
void atrousRecon(size_t &xdim, float *input, float* output, Param &par) {
    float SNR_THRESHOLD = par.reconSNR;
    int minScale = par.minScale;
    // int maxScale = par.maxScale;

    HaarWavelet haar;
    int numScales = haar.getNumScales(xdim);

    double *sigmafactors = (double*) malloc(sizeof(double) * (numScales + 1));
    for (int i = 0; i < numScales; i++) {
        sigmafactors[i] = haar.sigmaFactor(i);
    }

    float mean, originalSigma, oldSigma, newSigma;

    float *signal = (float*) malloc(sizeof(float) * xdim);

    for (int pos = 0; pos < xdim; pos++)
        output[pos] = 0;

    int filterW = haar.width() / 2;
    double *filter = (double*) malloc(sizeof(double) * haar.width());
    for (int i = 0; i < haar.width(); i++) {
        filter[i] = haar.coeff(i);
    }

    originalSigma = findMadfm(input, xdim);
    newSigma = 1.0e9;

    float threshold;
    int iter = 0;
    // while (iter < 100) {
    {
        oldSigma = newSigma;
        for (int i = 0; i < xdim; i++)
            signal[i] = input[i] - output[i];

        int spacing = 1;
        // This should be given to threads
        for (unsigned int scale = 1; scale <= numScales; scale++) {

            float *wavelet = (float*) malloc(sizeof(float) * xdim);
            for (size_t xpos = 0; xpos < xdim; xpos++) {
                wavelet[xpos] = signal[xpos];

                for (int xoffset = -filterW; xoffset <= filterW; xoffset++) {
                    long x = xpos + spacing * xoffset;

                    // Simplify this
                    while ((x < 0) || (x >= long(xdim))) {
                        if (x < 0)
                            x = -x;
                        else if (x >= long(xdim))
                            x = 2 * (xdim - 1) - x;
                    }

                    size_t filterpos = (xoffset + filterW);
                    size_t oldpos = x;

                    wavelet[xpos] -= filter[filterpos] * signal[oldpos];
                }
            }

            for (int pos = 0; pos < xdim; pos++) {
                signal[pos] = signal[pos] - wavelet[pos];
            }

            mean = findMedian(wavelet, xdim);

            threshold = mean + SNR_THRESHOLD * originalSigma * sigmafactors[scale];
            for (int pos = 0; pos < xdim; pos++) {
                output[pos] += wavelet[pos];
            }

            spacing *= 2;
            free(wavelet);
        }

        // for (int pos = 0; pos < xdim; pos++) {
        //     output[pos] += wavelet[pos];
        // }

        newSigma = findMadfmDiff(input, output, xdim);
    }

    free(filter);
    free(signal);
    free(sigmafactors);
}

int main() {
    int N = 10000;
    float *arr = (float*) malloc(sizeof(float) * N);
    float a = 5.0;

    srand(10);
    for (int i = 0; i < N; i++) {
        auto val = (float)std::rand() / (float)(RAND_MAX / a);
        arr[i] = val;
    }

    free(arr);

    return  0;
}
