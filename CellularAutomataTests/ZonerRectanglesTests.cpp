#include "stdafx.h"
#include "CppUnitTest.h"

#include <ZonerRectangles.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace ZonerTesting {
	TEST_CLASS(ZonerRectanglesTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		ZonerRectangles<int> deadZone{ 3,3 };
		deadZone.isLive(3, 4);
		deadZone.getDeadZones();
		deadZone.updateDeadZones(std::vector<std::vector<int>>(),std::vector<std::vector<int>>());
		Assert::IsTrue(true);
	}
	};
}	