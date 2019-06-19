#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/RulesConway.h"
#include "../CellularAutomata/SimulatorCPU.h"
#include "../CellularAutomata/SimulatorSequential.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SimulatorTesting {
	TEST_CLASS(SimulatorSequentialTesting) {
	public:
		TEST_METHOD(CanInstantiate) {
			RulesConway con = RulesConway();
			SimulatorCPU sim{6,4, con};
			Assert::IsTrue(true);
		}
	};
}