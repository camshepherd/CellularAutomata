#include "stdafx.h"
#include "CppUnitTest.h"

#include <Stopwatch.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
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