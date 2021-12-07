#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestAssert.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/XmlOutputter.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TextTestRunner.h>
#include <fstream>

#include "tdigest.cc"

class TestCentroid : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestCentroid);
  CPPUNIT_TEST(testCentroidCreation);
  CPPUNIT_TEST(testCentroidAddition);
  CPPUNIT_TEST_SUITE_END();

protected:
  void testCentroidCreation() {
    Centroid centroid(15.0, 1.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, centroid.mean(), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, centroid.weight(), 0.001);
  }

  void testCentroidAddition() {
    Centroid c1(15.0, 1.0);
    Centroid c2(5.0, 1.0);

    c1.add(c2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, c1.mean(), 0.001);
  }
};

static double quantile(const double q, const std::vector<double> &values) {
  double q1;
  if (values.size() == 0) {
    q1 = NAN;
  } else if (q == 1 || values.size() == 1) {
    q1 = values[values.size() - 1];
  } else {
    auto index = q * values.size();
    if (index < 0.5) {
      q1 = values[0];
    } else if (values.size() - index < 0.5) {
      q1 = values[values.size() - 1];
    } else {
      index -= 0.5;
      const int intIndex = static_cast<int>(index);
      q1 = values[intIndex + 1] * (index - intIndex) +
           values[intIndex] * (intIndex + 1 - index);
    }
  }
  return q1;
}

class TestTDigest : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestTDigest);
  CPPUNIT_TEST(testDigestInitialState);
  CPPUNIT_TEST(testDigestSingleValue);
  CPPUNIT_TEST(testDigestTwoValues);
  CPPUNIT_TEST(testDigestThreeValues);
  CPPUNIT_TEST(testDigestNormalCase);
  CPPUNIT_TEST(testDigestLargeArray);
  CPPUNIT_TEST_SUITE_END();

protected:
  void testDigestInitialState() {
    TDigest tdigest(100);
    CPPUNIT_ASSERT_EQUAL(0UL, tdigest.processed.size());
  }

  void testDigestSingleValue() {
    TDigest tdigest(100);
    tdigest.add(10.0);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, tdigest.quantile(0.0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, tdigest.quantile(0.5), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, tdigest.quantile(1.0), 0.001);
  }

  void testDigestTwoValues() {
    TDigest tdigest(100);
    tdigest.add(10.0);
    tdigest.add(20.0);

    tdigest.process();

    CPPUNIT_ASSERT_EQUAL(2UL, tdigest.processed.size());

    CPPUNIT_ASSERT_EQUAL(10.0, tdigest.processed[0].mean());
    CPPUNIT_ASSERT_EQUAL(20.0, tdigest.processed[1].mean());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, tdigest.quantile(0.0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, tdigest.quantile(0.5), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, tdigest.quantile(1.0), 0.001);
  }

  void testDigestThreeValues() {
    TDigest tdigest(100);
    tdigest.add(10.0);
    tdigest.add(20.0);
    tdigest.add(30.0);

    // CPPUNIT_ASSERT_EQUAL(2UL, tdigest.processed.size());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, tdigest.quantile(0.0), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, tdigest.quantile(0.5), 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(30.0, tdigest.quantile(1.0), 0.001);
  }

  void testDigestNormalCase() {
    int N = 90;
    std::uniform_real_distribution<> reals(0.0, 1.0);
    std::random_device gen;

    TDigest digest(100);

    std::vector<double> values;
    for (int i = 0; i < N; i++) {
      auto value = reals(gen);
      values.push_back(value);
      digest.add(value);
    }
    digest.compress();

    std::sort(values.begin(), values.end());
    auto quantile_median = quantile(0.5, values);

    auto digest_median = digest.quantile(0.5);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(quantile_median, digest_median, 0.1);
  }

  void testDigestLargeArray() {
    int N = 10000;
    std::uniform_real_distribution<> reals(0.0, 1.0);
    std::random_device gen;

    TDigest digest(100);

    std::vector<double> values;
    for (int i = 0; i < N; i++) {
      auto value = reals(gen);
      values.push_back(value);
      digest.add(value);
    }
    digest.compress();

    std::sort(values.begin(), values.end());
    auto quantile_median = quantile(0.5, values);
    auto digest_median = digest.quantile(0.5);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(quantile_median, digest_median, 0.01);
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCentroid);
CPPUNIT_TEST_SUITE_REGISTRATION(TestTDigest);

int main() {
  //   log4cxx::BasicConfigurator::configure();
  CppUnit::TestResult testResult;
  CppUnit::TestResultCollector collectedResult;
  CppUnit::BriefTestProgressListener progress;
  CppUnit::TestRunner runner;

  testResult.addListener(&collectedResult);
  testResult.addListener(&progress);

  runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
  runner.run(testResult);

  CppUnit::CompilerOutputter outputter(&collectedResult, std::cerr);
  outputter.write();

  std::ofstream out("result.xml");
  CppUnit::XmlOutputter xmlOutput(&collectedResult, out);
  xmlOutput.write();

  return collectedResult.wasSuccessful() ? 0 : 1;
}
