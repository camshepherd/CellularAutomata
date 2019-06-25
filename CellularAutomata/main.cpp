#include "SimulatorSequential.h"
#include "SimulatorCPU.h"

#include "RulesConway.h"
#include "RulesBML.h"

#include "SegmenterStrips.h"
#include "Stopwatch.h"

#include "ZonerPixels.h"
#include "ZonerRectangles.h"

#include <iostream>


int main() {
	std::cout << "The system compiles!" << std::endl;
	
	RulesConway rules{};
	SegmenterStrips strips{ 0 };
	SimulatorCPU cpu{5, 3, rules, strips};

	SimulatorSequential seq{ 90,4,rules };

	double cpuTime = cpu.stepForwardTime(3);
	double seqTime = seq.stepForwardTime(3);
	std::cout << "Has " << cpu.getNumFrames() << std::endl;
	//cpu.writeData("cpuOutput.txt");
	//seq.writeData("seqOutput.txt");

	std::cout << "Sequential: " << seqTime << ", Parallelised: " << cpuTime << std::endl;

	getchar();
}
