#include "stdafx.h"

#include "stdafx.h"
#include "CppUnitTest.h"

#include "RulesConway.h"
#include "SimulatorCPUZoning.h"
#include "SimulatorSequential.h"
#include "SegmenterStrips.h"
#include "ZonerPixels.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace SimulatorTesting{
	TEST_CLASS(SimulatorCPUZoningTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		RulesConway<int> con{};
		ZonerPixels<int> zoner{ 8,3 };
		SegmenterStrips seg{};
		SimulatorCPUZoning<int> sim{ 8,3,con, seg, zoner };
		sim.stepForward();
		sim.stepForwardTime(1.2f);
		Assert::IsTrue(true);
	}

	TEST_METHOD(StepsSingle) {
		RulesConway<int> con{};
		ZonerPixels<int> zoner{ 7,5 };
		SegmenterStrips seg{};
		SimulatorCPUZoning<int> sim{ 7,5, con, seg, zoner };
		SimulatorSequential<int> comp{ 7,5, con };

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
		RulesConway<int> con{};
		ZonerPixels<int> zoner{ 9,11 };
		SegmenterStrips seg{};
		SimulatorCPUZoning<int> sim{ 9,11, con, seg, zoner };
		SimulatorSequential<int> comp{ 9,11, con };

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