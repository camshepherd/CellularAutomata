#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/Zoner.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SimulatorTesting {
	TEST_CLASS(SimulatorCPUTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		Zoner deadZone{ 3,3 };
		deadZone.isLive(3, 4);
		deadZone.getDeadZones();
		Assert::IsTrue(true);
	}
	};
}