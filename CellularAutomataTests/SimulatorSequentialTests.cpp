#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/RulesConway.h"
#include "../CellularAutomata/SimulatorSequential.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SimulatorTesting {
	TEST_CLASS(SimulatorSequentialTesting) {
	public:
		TEST_METHOD(CanInstantiate) {
			RulesConway con = RulesConway();
			SimulatorSequential sim = SimulatorSequential(5,5,con);
			Assert::IsTrue(true);


			// nonSquare
			SimulatorSequential sim2 = SimulatorSequential(4, 8, con);
			Assert::IsTrue(true);
		}

		TEST_METHOD(CanCreateFrames) {
			RulesConway con = RulesConway();
			SimulatorSequential sim = SimulatorSequential(5, 5, con);
			Assert::AreEqual(sim.getNumFrames(), 1);

			sim.blankFrame();
			Assert::AreEqual(sim.getNumFrames(), 2);

			sim.blankFrame();
			Assert::AreEqual(sim.getNumFrames(), 3);

			//Test on non-square simulation
			SimulatorSequential sim2 = SimulatorSequential(4, 5, con);
			Assert::AreEqual(sim2.getNumFrames(), 1);

			sim2.blankFrame();
			Assert::AreEqual(sim2.getNumFrames(), 2);

			sim2.blankFrame();
			Assert::AreEqual(sim2.getNumFrames(), 3);
		}

		TEST_METHOD(CanEditCells) {
			RulesConway con = RulesConway();
			SimulatorSequential sim = SimulatorSequential(5, 5, con);

			Assert::AreEqual(sim.getCell(2, 2),0);
			Assert::AreEqual(sim.getCell(2, 2, 0), 0);

			sim.setCell(2, 2, 1);
			Assert::AreEqual(sim.getCell(2, 2), 1);
			Assert::AreEqual(sim.getCell(2, 2, 0), 1);
			Assert::AreEqual(sim.getCell(2, 1), 0);
			Assert::AreEqual(sim.getCell(2, 1, 0), 0);

			sim.setCell(3, 0, 1, 0);
			Assert::AreEqual(sim.getCell(3, 0), 1);
			
			sim.setCell(3, 0, 0, 0);
			Assert::AreEqual(sim.getCell(3, 0), 0);


			sim.blankFrame();
			sim.setCell(4, 3, 1);
			Assert::AreEqual(sim.getCell(4, 3), 1);
			Assert::AreEqual(sim.getCell(4, 3, 1), 1);

			// rectangular simulation
			SimulatorSequential sim2 = SimulatorSequential(2,5, con);
			Assert::AreEqual(sim2.getCell(1, 2), 0);
			Assert::AreEqual(sim2.getCell(1, 2, 0), 0);

			sim2.setCell(1, 4, 1);
			Assert::AreEqual(sim2.getCell(1,4), 1);
			Assert::AreEqual(sim2.getCell(1,4, 0), 1);
			Assert::AreEqual(sim2.getCell(1,3), 0);
			Assert::AreEqual(sim2.getCell(1,3, 0), 0);

		}

		TEST_METHOD(CanClearCells) {
			RulesConway con = RulesConway();
			SimulatorSequential sim = SimulatorSequential(5, 5, con);
			sim.blankFrame();
			sim.blankFrame();
			sim.blankFrame();
			sim.setCell(4, 2, 1);

			//clear, with blank frame
			sim.clear(true);

			Assert::AreEqual(sim.getNumFrames(), 1);

			// clear without blank frame
			sim.clear(false);
			Assert::AreEqual(sim.getNumFrames(), 0);


			// make sure that the state of the cells store is still healthy
			sim.blankFrame();
			Assert::AreEqual(sim.getNumFrames(), 1);
			


			// test with non-square board
			SimulatorSequential sim2 = SimulatorSequential(6, 3, con);

			sim2.blankFrame();
			sim2.blankFrame();
			sim2.blankFrame();
			sim2.setCell(5, 2, 1);

			sim2.clear(true);

			Assert::AreEqual(sim2.getNumFrames(), 1);
		}

		TEST_METHOD(CanStepCorrectly) {
			RulesConway con = RulesConway();
			SimulatorSequential sim = SimulatorSequential(4,4, con);

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
	};
}