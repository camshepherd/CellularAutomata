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
#include <fstream>

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
	std::string log_suffix = "PC";

	std::vector<ISimulator*> sims;
	int ydim = 1200, xdim = 1200;
	float simTime = 3;
	int repeats = 3;

	sims.push_back(new SimulatorSequential{ ydim, xdim, rules });
	simNames[0] = "Sequential";

	sims.push_back(new SimulatorCPU{ ydim, xdim, rules, strips });
	simNames[1] = "CPU Parallelised";

	std::ofstream log{ "results_" + log_suffix + ".out" };

	int numFrames;
	float meanFrames;
	try {
		for (int r = 0; r < sims.size(); ++r) {
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
			log << simNames[r] << ", " << meanFrames << "\n";
			// TODO: append line to the output file
		}
		log.close();
	}
	catch (std::exception e) {
		log.close();
		std::cout << "Got an error: " << e.what() << std::endl;
		return 1;
	}
	getchar();
	return 0;
}
