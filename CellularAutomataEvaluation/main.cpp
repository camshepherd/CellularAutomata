#include "SimulatorVector.h"
#include "SimulatorSequential.h"
#include "SimulatorCPU.h"
#include "SimulatorSequentialZoning.h"
#include "SimulatorCPUZoning.h"

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

using namespace CellularAutomata;

bool initialiseFrame(ISimulator<int>& sim, float density) {
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

	int ydim = 800, xdim = 800;

	RulesConway<int> rules{};
	SegmenterStrips stripsHor{ 0 };
	SegmenterStrips stripsVer{ 1 };
	std::map<int, std::string> simNames{};
	std::string log_suffix = "PC";
	std::string ruleset = "Conway";
	ZonerPixels<int> zoner{ ydim,xdim };

	std::vector<ISimulator<int>*> sims;
	
	float simTime = 0.1;
	int repeats = 3;
	float density = 0.3;
	sims.push_back(new SimulatorSequential<int>{ ydim, xdim, rules });
	simNames[0] = "Sequential";

	sims.push_back(new SimulatorCPU<int>{ ydim, xdim, rules, stripsHor });
	simNames[1] = "CPU Parallelised Horizontal";

	sims.push_back(new SimulatorCPU<int>{ ydim, xdim, rules, stripsVer });
	simNames[2] = "CPU Parallelised Vertical";

	sims.push_back(new SimulatorSequentialZoning<int>(ydim, xdim, rules, zoner));
	simNames[3] = "Sequential with pixel zoning";

	sims.push_back(new SimulatorCPUZoning<int>{ ydim,xdim,rules,stripsHor,zoner });
	simNames[4] = "Parallelised horizontal segmentation with pixel zoning";

	sims.push_back(new SimulatorCPUZoning<int>{ ydim,xdim,rules,stripsVer,zoner });
	simNames[5] = "Parallelised vertical segmentation with pixel zoning";

	std::ofstream log{ "results_" + log_suffix + ".out" };
	log << "Simulator, Ruleset, Y Dimension, X Dimension, Density, Mean frames simulated \n";
	int numFrames;
	float meanFrames;
	try {
		for (int r = 0; r < sims.size(); ++r) {
			for (ydim = 10; ydim < 1000; ydim *= 10) {
				for (xdim = 10; xdim < 1000; xdim *= 10) {
					for (float density = 0; density < 1; density += 0.2) {
						numFrames = 0;
						for (int e = 0; e < repeats; ++e) {
							sims[r]->clear();
							initialiseFrame(*sims[r], density);
							sims[r]->stepForwardTime(simTime);
							numFrames += sims[r]->getNumFrames();
						}
						// clear up the space that the simulation is taking up
						sims[r]->clear();
						meanFrames = numFrames / repeats;
						std::cout << simNames[r] << ", " << meanFrames << std::endl;
						log << simNames[r] << ", " << ruleset << "," << ydim << "," << xdim << "," << density << "," << meanFrames << "\n";
					}
				}
			}
			
		}
		log.close();
	}
	catch (std::exception e) {
		log.close();
		std::cout << "Got an error: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Finished testing" << std::endl;
	getchar();
	return 0;
}
