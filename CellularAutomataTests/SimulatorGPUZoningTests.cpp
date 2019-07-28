#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesArrayConway.hpp>
#include <SimulatorGPUZoning.hpp>
#include <SimulatorSequential.hpp>
#include <SegmenterStrips.hpp>
#include <ZonerArrayPixels.hpp>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace SimulatorTesting {
	TEST_CLASS(SimulatorGPUZoningTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		RulesArrayConway<int> con{};
		SegmenterStrips strips{};
		ZonerArrayPixels<int> zoner{6, 4};
		SimulatorGPUZoning<int> sim{ 6,4, con, strips,zoner };
		Assert::IsTrue(true);
	}

	TEST_METHOD(CanStepForward) {
		RulesArrayConway<int> con{};
		SegmenterStrips seg{};
		ZonerArrayPixels<int> zoner{ 4,4 };
		SimulatorGPUZoning<int> sim{ 4, 4, con, seg,zoner,3,32 };

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

	};
}