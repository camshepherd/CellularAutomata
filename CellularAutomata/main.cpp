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
#include <map>

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

	std::map<int, std::string> simNames{};


	std::vector<ISimulator*> sims;
	int ydim = 1200, xdim = 1200;
	float simTime = 3;
	int repeats = 3;

	sims.push_back(new SimulatorSequential{ ydim, xdim, rules });
	simNames[0] = "Sequential";

	sims.push_back(new SimulatorCPU{ ydim, xdim, rules, strips });
	simNames[1] = "CPU Parallelised";

	int numFrames;
	float meanFrames;
	for (int r = 0; r < sims.size(); ++r){
		numFrames = 0;
		for (int e = 0; e < repeats; ++e) {
			sims[r]->clear();
			sims[r]->stepForwardTime(simTime);
			numFrames += sims[r]->getNumFrames();
		}
		// clear up the space that the simulation is taking up
		sims[r]->clear();
		meanFrames = numFrames / repeats;
		std::cout << simNames[r] << ", " << meanFrames << std::endl;
		// TODO: append line to the output file
	}
	
	getchar();
}
