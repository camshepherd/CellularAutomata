#include "SimulatorSequential.h"
#include "SimulatorCPU.h"

#include "RulesConway.h"
#include "RulesBML.h"

#include "SegmenterStrips.h"
#include "Stopwatch.h"
#include <iostream>


int main() {
	std::cout << "The system compiles!" << std::endl;
	
	RulesConway rules{};
	SegmenterStrips strips{ 0 };
	SimulatorCPU cpu{5, 3, rules, strips};

	cpu.stepForward();
	cpu.stepForward();
	cpu.stepForward(2);
	std::cout << "Has " << cpu.getNumFrames() << std::endl;
	getchar();
}
