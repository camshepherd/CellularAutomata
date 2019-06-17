#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/RulesConway.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CellularAutomataTests
{		
	TEST_CLASS(RulesConwayTesting)
	{
	public:
		
		TEST_METHOD(CanInstantiateClass)
		{
			RulesConway thing = RulesConway();
			Assert::AreEqual(1, 1);
			Assert::IsTrue(thing.isValid(1));
		}

	};
}