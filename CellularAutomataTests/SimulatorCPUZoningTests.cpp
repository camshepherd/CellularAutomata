#include "stdafx.h"

#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesConway.hpp>
#include <SimulatorCPUZoning.hpp>
#include <SimulatorSequential.hpp>
#include <SegmenterStrips.hpp>
#include <ZonerPixels.hpp>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;

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

	TEST_METHOD(StepForwardMultiple)
	{
		RulesConway<int> con = RulesConway<int>();
		SegmenterStrips seg{};
		ZonerPixels<int> zoner{4, 4};
		SimulatorCPUZoning<int> sim{ 4, 4, con, seg,zoner };
		SimulatorSequential<int> refSim{ 4,4,con };
		refSim.stepForward(5);
		sim.stepForward(5);


		for (int y = 0; y < 4; ++y)
		{
			for (int x = 0; x < 4; ++x)
			{
				Assert::AreEqual(refSim.getCell(y, x), sim.getCell(y, x));
			}
		}
	}

	TEST_METHOD(CanResize) {
		RulesConway<int> bml{};
		SegmenterStrips seg{};
		SimulatorCPU<int> sim{ 19,19,bml,seg };
		Assert::AreEqual(sim.getYDim(), 19);
		Assert::AreEqual(sim.getXDim(), 19);

		sim.stepForward(20);

		sim.setDimensions(9, 4);
		Assert::AreEqual(sim.getYDim(), 9);
		Assert::AreEqual(sim.getXDim(), 4);

		sim.stepForward(20);


		// If this runs it should be fine
	}

};
}