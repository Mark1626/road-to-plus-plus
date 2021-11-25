#include <casacore/scimath/Functionals/Gaussian2D.h>
#include <casacore/scimath/Mathematics/MathFunc.h>
#include <iostream>

using casacore::GaussianConv;
using casacore::Float;
using std::cout;
using std::endl;

int main() {
    GaussianConv<Float> gaussianConv(0.5f, 1.0f);

    cout << "Support width: " << gaussianConv.sup_value() << endl;
    cout << gaussianConv.value(1.2f) << " " << gaussianConv.value(1.6f) << endl;
}
