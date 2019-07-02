#include "SimulatorVector.h"
#include "SimulatorSequential.h"
#include "SimulatorCPU.h"

#include "RulesConway.h"
#include "RulesBML.h"

#include "SegmenterStrips.h"
#include "Stopwatch.h"

#include "ZonerPixels.h"
#include "ZonerRectangles.h"

#include <iostream>
#include <memory>

bool initialiseFrame(ISimulator& sim, float density) {
	// density is the proportion of cells that start non-empty
	int xdim = sim.getXDim();
	int ydim = sim.getYDim();
	int maxVal = sim.getMaxValidState();
	int numCells = xdim * ydim;
	int numPopulated = static_cast<int>(numCells * density);

	for (int k = 0; k < numPopulated; ++k) {
		int y = std::rand() % ydim;
		int x = std::rand() % xdim;
		sim.setCell(y, x, rand() % (maxVal + 1));
	}
	return true;
}



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

	seq.clear();
	initialiseFrame(seq,0.3);
	
	int liveCount = 0, deadCount = 0;
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 3; ++x) {
			if (seq.getCell(y, x)) {
				++liveCount;
			}
			else {
				++deadCount;
			}
		}
	}

	std::cout << "There are " << liveCount << " live cells and " << deadCount << " dead cells!";

	getchar();
}
