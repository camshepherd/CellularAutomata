#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesArrayConway.hpp>
#include <SimulatorGPUZoning.hpp>
#include <SimulatorSequential.hpp>
#include <SegmenterStrips.hpp>
#include <ZonerArrayPixels.hpp>
#include "RulesConway.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace SimulatorTesting {
	TEST_CLASS(SimulatorGPUZoningTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		RulesArrayConway<int> con{};
		SegmenterStrips strips{};
		SimulatorGPUZoning<int> sim{ 6,4, con, strips,2,32};
		Assert::IsTrue(true);
	}

	TEST_METHOD(CanStepForward) {
		RulesArrayConway<int> con{};
		SegmenterStrips seg{};
		SimulatorGPUZoning<int> sim{ 4, 4, con, seg,3,32 };

		sim.stepForward();
		Assert::AreEqual(sim.getNumFrames(), 2);
		for (int y = 0; y < 4; ++y) {
			for (int x = 0; x < 4; ++x) {
				Assert::AreEqual(sim.getCell(y, x), 0);
			}
		}

		sim.setCell(2, 2, 1);
		sim.setCell(2, 1, 1);
		sim.setCell(1, 2, 1);
		sim.stepForward();
		Assert::AreEqual(sim.getCell(1, 1), 1);
		Assert::AreEqual(sim.getCell(2, 1), 1);
		Assert::AreEqual(sim.getCell(0, 0), 0);

		sim.setCell(0, 1, 1);
		sim.stepForward(1);
		Assert::AreEqual(sim.getCell(1, 1), 0);
		Assert::AreEqual(sim.getCell(0, 1), 1);

		Assert::AreEqual(sim.getNumFrames(), 4);
		const int numFrames = 3;
		sim.stepForward(numFrames);
		Assert::AreEqual(sim.getNumFrames(), 7);
	}

	TEST_METHOD(StepForwardMultiple)
	{
		RulesArrayConway<int> con = RulesArrayConway<int>();
		RulesConway<int> con2{};
		SegmenterStrips seg{};
		SimulatorGPUZoning<int> sim{ 4, 4, con, seg,2,32};
		SimulatorSequential<int> refSim{ 4,4,con2 };
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

	};
}