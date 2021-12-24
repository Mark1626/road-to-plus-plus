#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/aipstype.h>
#include <casacore/coordinates/Coordinates/CoordinateSystem.h>
#include <casacore/images/Images/FITSImage.h>

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Enter an argument " << std::endl << "Enter FITS to read";
        return 1;
    }

    // std::string fitsfile = "image.i.section.fits";
    // std::string fitsfile = "M33.image.fits";
    std::string fitsfile = argv[1];

    casacore::FITSImage img(fitsfile);
    const casacore::IPosition shape = img.shape();

    auto naxis = img.ndim();

    std::cout << "Shape " << shape << std::endl;
    std::cout << "Number of axes: " << naxis << std::endl;

    casacore::CoordinateSystem coor = img.coordinates();

    casacore::Vector<casacore::Double> world(naxis), pixel(naxis);
    pixel(0) = 32;
    pixel(1) = 32;
    pixel(2) = 0;
    // pixel(3) = 0;

    coor.toWorld(world, pixel);
    std::cout << "Pixel " << pixel << " in world " << world << std::endl;

    std::cout << "Coordinate System - Ref Pixel: " << coor.referencePixel() << std::endl;

    casacore::IPosition bottomleft(shape.nelements(), 0);
    casacore::IPosition topright(shape);
    topright -= 1;

    casacore::Slicer slicer(bottomleft, topright, casacore::Slicer::endIsLast);
    casacore::Array<casacore::Float> buffer;

    std::cout << "Reading slice " << slicer << std::endl;
    img.doGetSlice(buffer, slicer);

    std::cout << buffer(casacore::IPosition(3, 0, 0, 0)) << std::endl;

    return 0;
}
