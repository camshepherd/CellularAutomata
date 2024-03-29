#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesArrayBML.hpp>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace RulesTesting
{
	TEST_CLASS(RulesArrayBMLTesting)
	{
	public:

		TEST_METHOD(CanInstantiate)
		{
			RulesArrayBML<int> thing{};
			Assert::AreEqual(1, 1);
			Assert::IsTrue(thing.isValid(1));
		}

		TEST_METHOD(HandlesStateValidity) {
			RulesArrayBML<int> bml{};
			Assert::IsTrue(bml.isValid(0));
			Assert::IsTrue(bml.isValid(1));
			Assert::IsTrue(bml.isValid(2));
			Assert::IsFalse(bml.isValid(-1));
			Assert::IsFalse(bml.isValid(3));
		}

		TEST_METHOD(AppliesStepRule) {
			int* frame = static_cast<int*>(malloc(sizeof(int) * 9));
			for (int k = 0; k < 9; ++k) {
				frame[k] = 0;
			}

			//using default Conway rules: stay alive with 2 or 3 neighbours, be born if dead and have three neighbours
			RulesArrayBML<int> bml{};

			// make sure that the test frame was created correctly
			Assert::AreEqual(frame[0], 0);
			Assert::AreEqual(frame[8], 0);

			//check right-mover
			frame[4] = 1;
			Assert::AreEqual(bml.getNextState(frame, 1, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 1);
			Assert::AreEqual(bml.getNextState(frame, 1, 0), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 0), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			frame[4] = 0;

			//check up-mover
			frame[4] = 2;
			Assert::AreEqual(bml.getNextState(frame, 1, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 2);
			Assert::AreEqual(bml.getNextState(frame, 1, 0), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 0);
			frame[4] = 0;


			// check wrap-around right-mover
			frame[8] = 1;
			Assert::AreEqual(bml.getNextState(frame, 2, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 0), 1);
			Assert::AreEqual(bml.getNextState(frame, 0, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			frame[8] = 0;

			//check wrap-around down-mover
			frame[8] = 2;
			Assert::AreEqual(bml.getNextState(frame, 2, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 2), 2);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 1, 2), 0);
			Assert::AreEqual(bml.getNextState(frame, 2, 0), 0);
			frame[8] = 0;


			// check blocking
			frame[8] = 1;
			frame[6] = 2;
			Assert::AreEqual(bml.getNextState(frame, 2, 2), 1);
			Assert::AreEqual(bml.getNextState(frame, 2, 0), 0);
			frame[8] = 0;
			frame[6] = 0;


			// check moving out of the way
			frame[1] = 1;
			frame[7] = 2;

			Assert::AreEqual(bml.getNextState(frame, 0, 1), 2);
			Assert::AreEqual(bml.getNextState(frame, 2, 1), 0);
			Assert::AreEqual(bml.getNextState(frame, 0, 2), 1);
		}
	};
}