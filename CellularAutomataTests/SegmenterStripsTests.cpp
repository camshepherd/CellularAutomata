#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/SegmenterStrips.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SegmenterTesting {
	TEST_CLASS(SegmenterStripsTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		SegmenterStrips thing{ 1 };
		SegmenterStrips thing{ 0 };
		Assert::IsTrue(true);
	}
	};
}