#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/ZonerPixels.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ZonerTesting {
	TEST_CLASS(ZonerPixelsTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		ZonerPixels deadZone{ 3,3 };
		deadZone.isLive(3, 4);
		deadZone.updateDeadZones(std::vector<std::vector<std::vector<int>>>());
		Assert::IsTrue(true);
	}
	};
}
