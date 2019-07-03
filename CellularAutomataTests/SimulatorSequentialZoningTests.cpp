#include "stdafx.h"
#include "CppUnitTest.h"

#include "SimulatorSequentialZoning.h"
#include "RulesConway.h"
#include "ZonerPixels.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SimulatorTesting {
	TEST_CLASS(SimulatorSequentialZoningTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		RulesConway con{};
		ZonerPixels zoner{ 8,3 };
		SimulatorSequentialZoning sim{ 8,3,con,zoner };
		sim.stepForward();
		sim.stepForwardTime(1.2f);
		Assert::IsTrue(true);
	}

	TEST_METHOD(StepsSingle) {
		RulesConway con{};
		ZonerPixels zoner{ 7,5 };
		SimulatorSequentialZoning sim{ 7,5, con, zoner};
		SimulatorSequential comp{ 7,5, con};

		sim.setCell(1, 1, 1);
		comp.setCell(1, 1, 1);
		sim.stepForward();
		comp.stepForward();

		for (int y = 0; y < 7; ++y) {
			for (int x = 0; x < 5; ++x) {
				Assert::IsTrue(sim.getCell(y, x) == comp.getCell(y, x));
			}
		}
	}

	TEST_METHOD(StepsMany) {
		RulesConway con{};
		ZonerPixels zoner{ 9,11 };
		SimulatorSequentialZoning sim{ 9,11, con, zoner };
		SimulatorSequential comp{ 9,11, con };

		sim.setCell(1, 1, 1);
		comp.setCell(1, 1, 1);
		sim.stepForward(61);
		comp.stepForward(61);

		for (int y = 0; y < 9; ++y) {
			for (int x = 0; x < 11; ++x) {
				Assert::IsTrue(sim.getCell(y, x) == comp.getCell(y, x));
			}
		}
	}
	};
}