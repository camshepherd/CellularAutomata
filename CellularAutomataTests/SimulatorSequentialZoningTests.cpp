#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/SimulatorSequentialZoning.h"
#include "../CellularAutomata/RulesConway.h"
#include "../CellularAutomata/ZonerPixels.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SimulatorTesting {
	TEST_CLASS(SimulatorSequentialZoningTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		RulesConway con{};
		ZonerPixels zoner{ 8,3 };
		SimulatorSequentialZoning sim{ 8,3,con,zoner };
		sim.stepForward();
		Assert::IsTrue(true);
	}

	
	};
}