#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/casa/Quanta/MVTime.h>
#include <casacore/coordinates/Coordinates/Coordinate.h>
#include <casacore/coordinates/Coordinates/CoordinateSystem.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/coordinates/Coordinates/StokesCoordinate.h>
#include <casacore/fits/FITS/FITSDateUtil.h>
#include <casacore/fits/FITS/FITSKeywordUtil.h>
#include <casacore/fits/FITS/FITSReader.h>
#include <casacore/fits/FITS/fits.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MFrequency.h>
#include <fitsio.h>
#include <fstream>
#include <longnam.h>
#include <stdexcept>
#include <string>

namespace casa = casacore;

class FITSImageW {
  static const int BITPIX = -32;
  float minPix;
  float maxPix;
  casa::FitsKeywordList keywordList;
  std::string name;
  casa::IPosition shape;

public:
  FITSImageW(std::string name, casa::IPosition shape)
      : name(name), shape(shape) {}

  bool create(casa::CoordinateSystem csys) {
    std::ofstream outfile(name);
    if (!outfile.is_open()) {
      throw std::runtime_error("Unable to create file");
    }

    auto ndim = shape.nelements();

    casa::Record header;
    casa::Double b_scale, b_zero;
    // if (BITPIX == -32) {
    b_scale = 1.0;
    b_zero = 0.0;
    header.define("bitpix", BITPIX);
    header.setComment("bitpix", "Floating point (32 bit)");
    // }

    casa::Vector<casa::Int> naxis(ndim);
    for (int i = 0; i < ndim; i++)
      naxis(i) = shape(i);

    header.define("naxis", naxis);

    header.define("bscale", b_scale);
    header.setComment("bscale", "PHYSICAL = PIXEL * BSCALE + BZERO");
    header.define("bzero", b_zero);

    header.define("COMMENT1", "");

    header.define("BUNIT", "Jy");
    header.setComment("BUNIT", "Brightness (pixel) unit");

    casa::IPosition shapeCpy = shape;
    casa::CoordinateSystem wcs_copy = csys;

    casa::Record saveHeader(header);
    bool res = wcs_copy.toFITSHeader(header, shapeCpy, casa::True);

    // if (!res) {

    // }

    if (naxis.nelements() != shapeCpy.nelements()) {
      naxis.resize(shapeCpy.nelements());
      for (int idx = 0; idx < shapeCpy.nelements(); idx++)
        naxis(idx) = shapeCpy(idx);
      header.define("NAXIS", naxis);
    }

    casa::String date, timesys;
    casa::Time nowtime;
    casa::MVTime now(nowtime);
    casa::FITSDateUtil::toFITS(date, timesys, now);
    header.define("date", date);
    header.setComment("date", "Date FITS file was created");
    if (!header.isDefined("timesys") && !header.isDefined("TIMESYS")) {
      header.define("timesys", timesys);
      header.setComment("timesys", "Time system for HDU");
    }

    header.define("ORIGIN", "Simulation");
    keywordList = casa::FITSKeywordUtil::makeKeywordList();
    casa::FITSKeywordUtil::addKeywords(keywordList, header);

    keywordList.end();
    keywordList.first();
    keywordList.next();
    casa::FitsKeyCardTranslator keycard;
    const size_t card_size = 2880 * 4;
    char cards[card_size];
    memset(cards, 0, sizeof(cards));

    while (keycard.build(cards, keywordList)) {
      outfile << cards;
      memset(cards, 0, sizeof(cards));
    }

    if (cards[0] != 0) {
      outfile << cards;
    }
    return true;
  }

  static void wrapError(int code, int &status) {
    if (code) {
      char status_str[FLEN_STATUS];
      fits_get_errstatus(status, status_str);

      throw std::runtime_error("Error " + std::string(status_str) + " " +
                               std::to_string(status));
    }
  }

  void setUnits(const std::string &units) {
    fitsfile *fptr;
    int status = 0;

    wrapError(fits_open_file(&fptr, name.c_str(), READWRITE, &status), status);

    wrapError(fits_update_key(fptr, TSTRING, "BUNIT", (void *)(units.c_str()),
                              "Brightness unit", &status),
              status);

    wrapError(fits_close_file(fptr, &status), status);
  }

  void setRestoringBeam(double bmaj, double bmin, double bpa) {
    fitsfile *fptr;
    int status = 0;
    double radtodeg = 180.0 / M_PI;

    wrapError(fits_open_file(&fptr, name.c_str(), READWRITE, &status), status);

    double value = radtodeg * bmaj;
    wrapError(fits_update_key(fptr, TDOUBLE, "BMAJ", &value,
                              "Restoring beam major axis", &status),
              status);

    value = radtodeg * bmin;
    wrapError(fits_update_key(fptr, TDOUBLE, "BMIN", &value,
                              "Restoring beam minor axis", &status),
              status);

    value = radtodeg * bpa;
    wrapError(fits_update_key(fptr, TDOUBLE, "BPA", &value,
                              "Restoring beam position angle", &status),
              status);

    wrapError(fits_update_key(fptr, TSTRING, "BTYPE", (void *)"Intensity", " ",
                              &status),
              status);

    wrapError(fits_close_file(fptr, &status), status);
  }

  bool write(casa::Array<float> &arr, casa::IPosition &where) {
    fitsfile *fptr;
    int status = 0;
    double radtodeg = 180.0 / M_PI;

    wrapError(fits_open_file(&fptr, name.c_str(), READWRITE, &status), status);

    int hdutype;
    wrapError(fits_movabs_hdu(fptr, 1, &hdutype, &status), status);

    int naxis;
    wrapError(fits_movabs_hdu(fptr, 1, &hdutype, &status), status);

    wrapError(fits_get_img_dim(fptr, &naxis, &status), status);

    long *axes = new long[naxis];
    wrapError(fits_get_img_size(fptr, naxis, axes, &status), status);

    long fpixel[4], lpixel[4];
    int array_dim = arr.shape().nelements();
    int location_dim = where.nelements();

    fpixel[0] = where[0] + 1;
    lpixel[0] = where[0] + arr.shape()[0];
    fpixel[1] = where[1] + 1;
    lpixel[1] = where[1] + arr.shape()[1];

    if (array_dim == 2 && location_dim >= 3) {
      fpixel[2] = where[2] + 1;
      lpixel[2] = where[2] + 1;
      if (location_dim == 4) {
        fpixel[3] = where[3] + 1;
        lpixel[3] = where[3] + 1;
      }
    } else if (array_dim == 3 && location_dim >= 3) {
      fpixel[2] = where[2] + 1;
      lpixel[2] = where[2] + arr.shape()[2];
      if (location_dim == 4) {
        fpixel[3] = where[3] + 1;
        lpixel[3] = where[3] + 1;
      }
    } else if (array_dim == 4 && location_dim == 4) {
      fpixel[2] = where[2] + 1;
      lpixel[2] = where[2] + arr.shape()[2];
      fpixel[3] = where[3] + 1;
      lpixel[3] = where[3] + arr.shape()[3];
    }

    int64_t nelements = arr.nelements();
    bool toDelete = false;
    const float *data = arr.getStorage(toDelete);
    float *dataptr = (float *)data;

    long group = 0;
    wrapError(fits_write_subset_flt(fptr, group, naxis, axes, fpixel, lpixel,
                                    dataptr, &status),
              status);

    wrapError(fits_close_file(fptr, &status), status);
    delete[] axes;
    return true;
  }
};

int main() {
  std::string name = "dummy.fits";
  int xsize = 128;
  int ysize = 128;
  int nchan = 48;

  casa::IPosition shape(2, xsize, ysize);
  casa::CoordinateSystem wcs;

  /*
    1 0
    0 1
  */
  casacore::Matrix<double> xform(2, 2);
  xform = 0.0;
  xform.diagonal() = 1.0;

  // RA-Dec Axis
  casa::DirectionCoordinate direction(
      casa::MDirection::J2000, casa::Projection(casa::Projection::SIN),
      294 * casa::C::pi / 180.0, -60 * casa::C::pi / 180,
      -0.001 * casa::C::pi / 180.0, 0.001 * casa::C::pi / 180.0, xform,
      xsize / 2.0, ysize / 2.0);

  casa::Vector<casa::String> units(2);
  units = "deg";
  direction.setWorldAxisUnits(units);

  // Spectral Axis
  casa::SpectralCoordinate spectral(casa::MFrequency::TOPO, 1400E+6, 20E+3, 0,
                                    1420.40575E+6);

  units.resize(1);
  units = "MHz";
  spectral.setWorldAxisUnits(units);

  // Polarisaiton Axis
  const casacore::Vector<casacore::Stokes::StokesTypes> stokesVec = std::vector<casacore::Stokes::StokesTypes>(1, casacore::Stokes::type("I"));


  wcs.addCoordinate(direction);
  wcs.addCoordinate(spectral);
  shape.append(casa::IPosition(1, nchan));

  casacore::StokesCoordinate stokes(std::vector<int>(1, 1));
  wcs.addCoordinate(stokes);


  // Pixels
  casa::Matrix<casa::Float> pixels(xsize, ysize);
  pixels = 0.0;
  pixels.diagonal() = 0.001;
  pixels.diagonal(2) = 0.002;

  casa::IPosition where(2, 0, 0);

  FITSImageW fits(name, shape);
  fits.create(wcs);
  fits.setUnits("Jy/pixel");
  fits.setRestoringBeam(2.0e-4, 1.0e-4, 1.0e-1);
  fits.write(pixels, where);
}
