#include "stdafx.h"
#include "CppUnitTest.h"

#include "RulesConway.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace RulesTesting
{		
	TEST_CLASS(RulesConwayTesting)
	{
	public:
		
		TEST_METHOD(CanInstantiate)
		{
			RulesConway thing = RulesConway();
			Assert::AreEqual(1, 1);
			Assert::IsTrue(thing.isValid(1));
		}


		TEST_METHOD(HandlesStateValidity) {
			RulesConway con01(2, 3, 4, 5, 0, 1);
			RulesConway con02(3, 5, 4, 6, 2, 2);
			
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
			std::vector<std::vector<int>> frame(3, std::vector<int>(3,0));
			//using default Conway rules: stay alive with 2 or 3 neighbours, be born if dead and have three neighbours
			RulesConway con{};

			// make sure that the test frame was created correctly
			Assert::AreEqual(frame[0][0], 0);
			Assert::AreEqual(frame[2][2], 0);

			// all blank
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}

			frame[0][0] = 1;
			// single live cell
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}


			// Two live cells
			frame[1][1] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}

			// three live cells
			frame[2][1] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 1);
				}
			}

			// Four live cells
			frame[2][2] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					if (frame[y][x] == 0) {
						Assert::AreEqual(con.getNextState(frame, y, x), 0);
					}
					else {
						Assert::AreEqual(con.getNextState(frame, y, x), 1);
					}
				}
			}

			// five live cells
			frame[2][0] = 1;
			for (int y = 0; y < 3; ++y) {
				for (int x = 0; x < 3; ++x) {
					Assert::AreEqual(con.getNextState(frame, y, x), 0);
				}
			}
		}
	};
}