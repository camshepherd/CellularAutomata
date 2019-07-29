#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesConway.hpp>
#include <SimulatorCPU.hpp>
#include <SimulatorSequential.hpp>
#include <SegmenterStrips.hpp>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace SimulatorTesting {
	TEST_CLASS(SimulatorCPUTesting) {
	public:
		TEST_METHOD(CanInstantiate) {
			RulesConway<int> con = RulesConway<int>();
			SegmenterStrips strips{};
			SimulatorCPU<int> sim{6,4, con, strips};
			Assert::IsTrue(true);
		}

		TEST_METHOD(CanStepForward) {
			RulesConway<int> con = RulesConway<int>();
			SegmenterStrips seg{};
			SimulatorCPU<int> sim{ 4, 4, con, seg };

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
			int numFrames = 3;
			sim.stepForward(numFrames);
			Assert::AreEqual(sim.getNumFrames(), 7);
		}

		TEST_METHOD(StepForwardMultiple)
		{
			RulesConway<int> con = RulesConway<int>();
			SegmenterStrips seg{};
			SimulatorCPU<int> sim{ 4, 4, con, seg };
			SimulatorSequential<int> refSim{ 4,4,con };
			refSim.stepForward(5);
			sim.stepForward(5);


			for (int y = 0; y < 4; ++y)
			{
				for(int x = 0; x < 4; ++x)
				{
					Assert::AreEqual(refSim.getCell(y, x), sim.getCell(y, x));
				}
			}
		}
	};
}