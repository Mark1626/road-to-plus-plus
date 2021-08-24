#include <casacore/measures/Measures/MeasTable.h>
#include <casacore/measures/Measures/MPosition.h>
#include <iostream>
int main()
{
  casacore::MPosition mpos;
  casacore::MeasTable::Observatory(mpos, "DWL");
  std::cout << "DWL " << mpos << std::endl;
  return 0;
}
