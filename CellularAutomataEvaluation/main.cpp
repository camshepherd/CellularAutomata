#include "pch.h"
#include <iostream>
#include <memory>
#include <map>
#include <fstream>
#include "RulesArrayConway.hpp"
#include "SimulatorGPU.hpp"

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

// TODO: add frame initialisation for ISimulatorArrays.

int main() {
	std::cout << "The system compiles!" << std::endl;

	int ydim = 100, xdim = 100;

	RulesArrayConway<int> con2{};
	SegmenterStrips seg2{};
	
	
	SimulatorGPU<int> sim2{ 3,3,con2,seg2 };
	sim2.setLaunchParams(2, 2);
	for(int y = 0; y < 3; ++y)
	{
		std::cout << std::endl;
		for(int x = 0; x < 3; ++x)
		{
			sim2.setCell(y, x, 0);
		}
	}
	sim2.setCell(2, 2, 1);
	sim2.setCell(2, 1, 1);
	sim2.setCell(1, 2, 1);
	sim2.setCell(1, 1, 1);
	sim2.setCell(0, 1, 1);
	sim2.setCell(1, 0, 1);
	std::cout << "\nPrint out frame" << std::endl;
	for (int y = 0; y < 3; ++y)
	{
		std::cout << std::endl;
		for (int x = 0; x < 3; ++x)
		{
			std::cout << sim2.getCell(y, x);
		}
	}
	std::cout << std::endl;
	sim2.stepForward(1);
	
	std::cout << "\nPrinting stored frame in the simulator" << std::endl;
	for(int y = 0; y < 3; ++y)
	{
		std::cout << std::endl;
		for (int x = 0; x < 3; ++x)
		{
			std::cout << sim2.getCell(y, x);
		}
	}
	std::cout << std::endl;
	getchar();

	sim2.stepForward();
	
	for (int y = 0; y < 3; ++y)
	{
		std::cout << std::endl;
		for (int x = 0; x < 3; ++x)
		{
			std::cout << sim2.getCell(y, x);
		}
	}
	std::cout << std::endl;
	sim2.stepForward(1);
	for (int y = 0; y < 3; ++y)
	{
		std::cout << std::endl;
		for (int x = 0; x < 3; ++x)
		{
			std::cout << sim2.getCell(y, x);
		}
	}
	std::cout << "Finished initial stuff" << std::endl;
	RulesConway<int> rules{};
	SegmenterStrips stripsHor{ 0 };
	SegmenterStrips stripsVer{ 1 };
	std::map<int, std::string> simNames{};
	std::string ID;
	std::cout << "Please enter the Computer ID (single digit): " << std::endl;
	std::cin >> ID;
	std::string ruleset = "Conway";
	ZonerPixels<int> zoner{ ydim,xdim };

	std::vector<ISimulator<int>*> sims;
	
	float simTime = 0.1;
	int repeats = 3;
	float density = 0.3;
	sims.push_back(new SimulatorSequential<int>{ ydim, xdim, rules });
	simNames[0] = "Seq";

	sims.push_back(new SimulatorCPU<int>{ ydim, xdim, rules, stripsHor });
	simNames[1] = "CPUHor";

	sims.push_back(new SimulatorCPU<int>{ ydim, xdim, rules, stripsVer });
	simNames[2] = "CPUVer";

	sims.push_back(new SimulatorSequentialZoning<int>(ydim, xdim, rules, zoner));
	simNames[3] = "SeqZon";

	sims.push_back(new SimulatorCPUZoning<int>{ ydim,xdim,rules,stripsHor,zoner });
	simNames[4] = "CPUHorZon";

	sims.push_back(new SimulatorCPUZoning<int>{ ydim,xdim,rules,stripsVer,zoner });
	simNames[5] = "CPUVerZon";

	std::ofstream log{ "results_" + ID + ".out" };
	log << "Simulator,Ruleset,YDimension,XDimension,Density,MeanFramesSimulated\n";
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
