#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/RulesBML.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace RulesTesting
{
	TEST_CLASS(RulesBMLTesting)
	{
	public:

		TEST_METHOD(CanInstantiate)
		{
			RulesBML thing = RulesBML();
			Assert::AreEqual(1, 1);
			Assert::IsTrue(thing.isValid(1));
		}

		TEST_METHOD(HandlesStateValidity) {
			RulesBML bml = RulesBML();
			Assert::IsTrue(bml.isValid(0));
			Assert::IsTrue(bml.isValid(1));
			Assert::IsTrue(bml.isValid(2));
			Assert::IsFalse(bml.isValid(-1));
			Assert::IsFalse(bml.isValid(3));
		}

		TEST_METHOD(AppliesStepRule) {
			std::vector<std::vector<int>> frame(3, std::vector<int>(3, 0));
			//using default Conway rules: stay alive with 2 or 3 neighbours, be born if dead and have three neighbours
			RulesBML bml{};

			// make sure that the test frame was created correctly
			Assert::AreEqual(frame[0][0], 0);
			Assert::AreEqual(frame[2][2], 0);

			//check right-mover
			frame[1][1] = 1;
			Assert::AreEqual(bml.getNextState(frame, 1, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 1);
			Assert::AreEqual(bml.getNextState(frame, 1, 0), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 0), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			frame[1][1] = 0;

			//check up-mover
			frame[1][1] = 2;
			Assert::AreEqual(bml.getNextState(frame, 1, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 2);
			Assert::AreEqual(bml.getNextState(frame, 1, 0), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 0);
			frame[1][1] = 0;


			// check wrap-around right-mover
			frame[2][2] = 1;
			Assert::AreEqual(bml.getNextState(frame, 2, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 0), 1);
			Assert::AreEqual(bml.getNextState(frame, 0, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			frame[2][2] = 0;

			//check wrap-around down-mover
			frame[2][2] = 2;
			Assert::AreEqual(bml.getNextState(frame, 2, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 2), 2);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 0), 0);
			frame[2][2] = 0;


			// check blocking
			frame[2][2] = 1;
			frame[2][0] = 2;
			Assert::AreEqual(bml.getNextState(frame, 2, 2), 1);
			Assert::AreEqual(bml.getNextState(frame, 2, 0), 0);
			frame[2][2] = 0;
			frame[2][0] = 0;

			
			// check moving out of the way
			frame[0][1] = 1;
			frame[2][1] = 2;

			Assert::AreEqual(bml.getNextState(frame, 0, 1), 2);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 2), 1);
		}
	};
}