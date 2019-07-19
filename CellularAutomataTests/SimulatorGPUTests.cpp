#include "stdafx.h"
#include "CppUnitTest.h"

#include <RulesArrayConway.hpp>
#include <SimulatorGPU.hpp>
#include <SimulatorSequential.hpp>
#include <SegmenterStrips.hpp>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace SimulatorTesting {
	TEST_CLASS(SimulatorGPUTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		RulesArrayConway<int> con{};
		SegmenterStrips strips{};
		SimulatorGPU<int> sim{ 6,4, con, strips,2,32 };
		Assert::IsTrue(true);
	}

	};
}