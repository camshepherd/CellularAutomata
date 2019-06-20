#include "stdafx.h"
#include "CppUnitTest.h"

#include "../CellularAutomata/Stopwatch.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PeripheralTesting {
	TEST_CLASS(StopwatchTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		Stopwatch clock{};
		Assert::IsTrue(true);
	}

	TEST_METHOD(FunctionsDoNotBreak) {
		Stopwatch clock{};
		clock.reset();
		float elapsed = clock.elapsed();
		Assert::IsTrue(true);
	}
	};
}