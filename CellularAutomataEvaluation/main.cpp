#include "pch.h"
#include <iostream>
#include <memory>
#include <map>
#include <fstream>
#include "RulesArrayConway.hpp"
#include "SimulatorGPU.hpp"
#include "RulesArrayBML.hpp"
#include "ZonerArrayPixels.hpp"
#include "SimulatorGPUZoning.hpp"

using namespace CellularAutomata;

template <typename T>
bool initialiseFrame(ISimulator<T>& sim, float density) {
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
	
	// TODO: Test both BML and GoL!

	int ydim = 100, xdim = 100;
	int nBlocks = 2, nThreads = 16;
	SegmenterStrips stripsHor{ 0 };
	SegmenterStrips stripsVer{ 1 };
	int nSegments = 4;
	std::map<int, std::string> simNames{};
	std::string ID;
	int nFrames = 20;
	std::string ruleSet = "Conway";
	std::map<std::string, IRules<int>*> rules{};
	rules["Conway"] = new RulesConway<int>{};
	rules["BML"] = new RulesBML<int>{};
	std::map<std::string, IRulesArray<int>*>rulesArray{};
	rulesArray["Conway"] = new RulesArrayConway<int>{};
	rulesArray["BML"] = new RulesArrayBML<int>{};
	ZonerPixels<int> zoner{ ydim,xdim };
	ZonerArrayPixels<int> zoner2{ ydim,xdim };
	std::vector<ISimulator<int>*> sims;

	std::cout << "Please enter the Computer ID (single digit): " << std::endl;
	std::cin >> ID;
	float simTime = 0.1;
	int repeats = 3;
	
	float density = 0.3;
	sims.push_back(new SimulatorSequential<int>{ ydim, xdim, *rules[ruleSet] });
	simNames[0] = "Seq";

	sims.push_back(new SimulatorSequentialZoning<int>(ydim, xdim, *rules[ruleSet], zoner));
	simNames[1] = "SeqZon";

	sims.push_back(new SimulatorCPU<int>{ ydim, xdim, *rules[ruleSet], stripsHor });
	simNames[2] = "CPUHor";

	sims.push_back(new SimulatorCPU<int>{ ydim, xdim, *rules[ruleSet], stripsVer });
	simNames[3] = "CPUVer";

	sims.push_back(new SimulatorCPUZoning<int>{ ydim,xdim,*rules[ruleSet],stripsHor,zoner });
	simNames[4] = "CPUHorZon";

	sims.push_back(new SimulatorCPUZoning<int>{ ydim,xdim,*rules[ruleSet],stripsVer,zoner });
	simNames[5] = "CPUVerZon";

	sims.push_back(new SimulatorGPU<int>{ ydim,xdim,*rulesArray[ruleSet],stripsVer});
	simNames[6] = "GPUVer";

	sims.push_back(new SimulatorGPU<int>{ ydim,xdim,*rulesArray[ruleSet],stripsHor });
	simNames[7] = "GPUHor";

	sims.push_back(new SimulatorGPUZoning<int>{ ydim,xdim,*rulesArray[ruleSet],stripsVer,zoner2,nBlocks,nThreads });
	simNames[8] = "GPUVerZon";

	sims.push_back(new SimulatorGPUZoning<int>{ ydim,xdim,*rulesArray[ruleSet],stripsHor,zoner2,nBlocks,nThreads });
	simNames[9] = "GPUHorZon";

	std::ofstream log{ "results_" + ID + ".out" };
	log << "Simulator,Ruleset,YDimension,XDimension,Density,MeanTime,nBlocks,nThreads,nSegments\n";
	double totalTime = 0;
	double meanTime;
	//// Sequential
	try {
		for (int r = 0; r < 2; ++r) {
			for (ydim = 10; ydim < 1000; ydim *= 10) {
				for (xdim = 10; xdim < 1000; xdim *= 10) {
					for (density = 0; density < 1; density += 0.2) {
						totalTime = 0;

						for (int e = 0; e < repeats; ++e) {
							sims[r]->clear();
							initialiseFrame(*sims[r], density);
							totalTime += sims[r]->stepForward(nFrames);
						}
						// clear up the space that the simulation is taking up
						sims[r]->clear();
						meanTime = totalTime / repeats;
						std::cout << simNames[r] << ", " << meanTime << std::endl;
						log << simNames[r] << ", " << ruleSet << "," << ydim << "," << xdim << "," << density << "," << meanTime << "," << nBlocks << "," << nThreads << "," << "-1" << "," << "\n";
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

	// CPU Parallelised
	int params[3];
	try {
		for (int r = 2; r < 6; ++r) {
			for (ydim = 10; ydim < 1000; ydim *= 10) {
				for (xdim = 10; xdim < 1000; xdim *= 10) {
					for (float density = 0; density < 1; density += 0.2) {
						for (nSegments = 1; nSegments < 30; ++nSegments)
						{
							totalTime = 0;
							for (int e = 0; e < repeats; ++e) {
								sims[r]->clear();
								initialiseFrame(*sims[r], density);
								params[0] = nSegments;
								sims[r]->setParams(params);
								totalTime += sims[r]->stepForward(nFrames);
							}
							// clear up the space that the simulation is taking up
							sims[r]->clear();
							meanTime = totalTime / repeats;
							std::cout << simNames[r] << ", " << meanTime << std::endl;
							log << simNames[r] << ", " << ruleSet << "," << ydim << "," << xdim << "," << density << "," << meanTime << "," << nBlocks << "," << nThreads << "," << nSegments << "," << "\n";
						}
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

	// GPU Parallelised
	try {
		for (int r = 6; r < 10; ++r) {
			for (ydim = 10; ydim < 1000; ydim *= 10) {
				for (xdim = 10; xdim < 1000; xdim *= 10) {
					for (float density = 0; density < 1; density += 0.2) {
						
						for (nBlocks = 1; nBlocks < 8; ++nBlocks)
						{
							for (nThreads = 2; nThreads < 512; ++nThreads)
							{
								for (nSegments = 1; nSegments < nBlocks * nThreads * 2; ++nSegments)
								{
									totalTime = 0;
									for (int e = 0; e < repeats; ++e) {
										sims[r]->clear();
										initialiseFrame(*sims[r], density);
										params[0] = nSegments;
										params[1] = nBlocks;
										params[2] = nThreads;
										sims[r]->setParams(params);
										totalTime += sims[r]->stepForward(nFrames);
									}
									// clear up the space that the simulation is taking up
									sims[r]->clear();
									meanTime = totalTime / repeats;
									std::cout << simNames[r] << ", " << meanTime << std::endl;
									log << simNames[r] << ", " << ruleSet << "," << ydim << "," << xdim << "," << density << "," << meanTime << "," << nBlocks << "," << nThreads << "," << nSegments << "," << "\n";
								}
							}
						}
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
