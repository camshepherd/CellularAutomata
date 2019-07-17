#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesArrayConway.hpp>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace RulesTesting
{
	TEST_CLASS(RulesArrayConwayTesting)
	{
	public:

		TEST_METHOD(CanInstantiate)
		{
			RulesArrayConway<int> thing = RulesArrayConway<int>();
			RulesArrayConway<bool> thing2{};
			Assert::AreEqual(1, 1);
			Assert::IsTrue(thing.isValid(1));
		}

		TEST_METHOD(HandlesStateValidity) {
			RulesArrayConway<int> con01{ 2, 3, 4, 5, 0, 1, 2,2 };
			RulesArrayConway<int> con02{ 3, 5, 4, 6, 2, 2, 2,2 };

			Assert::IsTrue(con01.isValid(0));
			Assert::IsTrue(con01.isValid(1));
			Assert::IsFalse(con01.isValid(2));
			Assert::IsFalse(con01.isValid(-1));

			Assert::IsTrue(con02.isValid(2));

			Assert::IsFalse(con02.isValid(1));
			Assert::IsFalse(con02.isValid(0));
			Assert::IsFalse(con02.isValid(3));
		}

		TEST_METHOD(CorrectlyAppliesStepRuleWithWrapAround) {
			int* frame = static_cast<int*>(malloc(sizeof(int) * 9));
			for (int k = 0; k < 9; ++k) {
				frame[k] = 0;
			}
			//using default Conway rules: stay alive with 2 or 3 neighbours, be born if dead and have three neighbours
			RulesArrayConway<int> con{3,3};

			// make sure that the test frame was created correctly
			Assert::AreEqual(frame[0], 0);
			Assert::AreEqual(frame[8], 0);

			// all blank
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}

			frame[0] = 1;
			// single live cell
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}


			// Two live cells
			frame[4] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}

			// three live cells
			frame[7] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 1);
				}
			}

			// Four live cells
			frame[8] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					if (frame[y*3 + x] == 0) {
						Assert::AreEqual(con.getNextState(frame, y, x), 0);
					}
					else {
						Assert::AreEqual(con.getNextState(frame, y, x), 1);
					}
				}
			}

			// five live cells
			frame[6] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}
		}
		
	};
}