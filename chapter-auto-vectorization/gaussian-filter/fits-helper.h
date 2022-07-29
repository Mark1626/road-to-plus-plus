#pragma once
#include "fitsio.h"
#include <stdio.h>
#include <string.h>

typedef struct {
  float *data;
  long naxes[4];
} fits_data_t;

void destroy_fits(fits_data_t *fits) {
  if (fits->data) {
    free(fits->data);
  }
}

int extract_data_from_fits(fitsfile *fptr, fits_data_t *fits) {
  int hdutype, naxis;
  int status = 0;
  long *naxes = fits->naxes;
  long fpixel[4] = {1, 1, 1, 1};

  if (fits_get_hdu_type(fptr, &hdutype, &status) || hdutype != IMAGE_HDU) {
    printf("Error: this program only works on images, not tables\n");
    abort();
  }

  fits_get_img_dim(fptr, &naxis, &status);
  if (status || naxis != 4) {
    printf("Error: NAXIS = %d.  Only 4-D images are supported.\n", naxis);
    abort();
  }
  fits_get_img_size(fptr, 4, naxes, &status);
  if (status || naxes[2] != 1 || naxes[3] != 1) {

    printf("Error: Dim = %ld x %ld x %ld x %ld. Dimension of frequency and "
           "polarisation axis has to be 1.\n",
           naxes[0], naxes[1], naxes[2], naxes[3]);
    abort();
  }

  printf("Dim: %ld x %ld x %ld x %ld\n", naxes[0], naxes[1], naxes[2],
         naxes[3]);

  fits->data = (float *)malloc(naxes[0] * naxes[1] * sizeof(float));

  if (fits->data == NULL) {
    printf("Memory allocation error\n");
    abort();
  }

  fits_read_pix(fptr, TFLOAT, fpixel, naxes[0] * naxes[1], 0, fits->data, 0,
                &status);

  return status;
}

int store_data_copying_fits_header(fitsfile *infptr, const char *filename,
                                   fits_data_t *fits) {
  fitsfile *outfptr;
  int status = 0;
  long npixels;
  long first_pix[4] = {1, 1, 1, 1};

  if (!fits_create_file(&outfptr, filename, &status)) {
    fits_copy_header(infptr, outfptr, &status);
    npixels = fits->naxes[0] * fits->naxes[1];

    fits_write_pix(outfptr, TFLOAT, first_pix, npixels, fits->data, &status);
  }

  fits_close_file(outfptr, &status);

  return status;
}
