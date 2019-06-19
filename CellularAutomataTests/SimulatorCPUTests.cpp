#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/RulesConway.h"
#include "../CellularAutomata/SimulatorCPU.h"
#include "../CellularAutomata/SimulatorSequential.h"
#include "../CellularAutomata/SegmenterStrips.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SimulatorTesting {
	TEST_CLASS(SimulatorCPUTesting) {
	public:
		TEST_METHOD(CanInstantiate) {
			RulesConway con = RulesConway();
			SegmenterStrips strips{};
			SimulatorCPU sim{6,4, con, strips};
			Assert::IsTrue(true);
		}
	};
}