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
#include <cctype>

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

template <typename T>
bool runSimulations(std::ofstream& log, std::ofstream& error_log, int ydim, int xdim, int nFrames)
{

	int nBlocks = 2, nThreads = 16;
	SegmenterStrips stripsHor{ 0 };
	SegmenterStrips stripsVer{ 1 };
	int nSegments = 4;
	std::map<int, std::string> simNames{};

	std::string ruleSet/* = "Conway"*/;
	std::map<std::string, IRules<T>*> rules{};
	rules["Conway"] = new RulesConway<T>{};
	rules["BML"] = new RulesBML<T>{};
	std::map<std::string, IRulesArray<T>*>rulesArray{};
	rulesArray["Conway"] = new RulesArrayConway<T>{};
	rulesArray["BML"] = new RulesArrayBML<T>{};
	ZonerPixels<T> zoner{ ydim,xdim };
	std::vector<ISimulator<T>*> sims;

	
	float simTime = 0.1;
	int repeats = 3;

	float density = 0.3;
	double totalTime = 0;
	double meanTime;

	int params[5];

	for (typename std::map<std::string, IRules<T>*>::iterator it = rules.begin(); it != rules.end(); ++it) {
		ruleSet = it->first;
		//ruleSet = "Conway";
		sims.clear();
		sims.push_back(new SimulatorSequential<T>{ ydim, xdim, *rules[ruleSet] });
		simNames[0] = "Seq";

		sims.push_back(new SimulatorSequentialZoning<T>(ydim, xdim, *rules[ruleSet], zoner));
		simNames[1] = "SeqZon";

		sims.push_back(new SimulatorCPU<T>{ ydim, xdim, *rules[ruleSet], stripsHor });
		simNames[2] = "CPUHor";

		sims.push_back(new SimulatorCPU<T>{ ydim, xdim, *rules[ruleSet], stripsVer });
		simNames[3] = "CPUVer";

		sims.push_back(new SimulatorCPUZoning<T>{ ydim,xdim,*rules[ruleSet],stripsHor,zoner });
		simNames[4] = "CPUHorZon";

		sims.push_back(new SimulatorCPUZoning<T>{ ydim,xdim,*rules[ruleSet],stripsVer,zoner });
		simNames[5] = "CPUVerZon";

		sims.push_back(new SimulatorGPUZoning<T>{ ydim,xdim,*rulesArray[ruleSet],stripsVer,nBlocks,nThreads });
		simNames[6] = "GPUVerZon";

		sims.push_back(new SimulatorGPUZoning<T>{ ydim,xdim,*rulesArray[ruleSet],stripsHor,nBlocks,nThreads });
		simNames[7] = "GPUHorZon";

		sims.push_back(new SimulatorGPU<T>{ ydim,xdim,*rulesArray[ruleSet],stripsVer });
		simNames[8] = "GPUVer";

		sims.push_back(new SimulatorGPU<T>{ ydim,xdim,*rulesArray[ruleSet],stripsHor });
		simNames[9] = "GPUHor";

		


		// Sequential
		//try {
		//	for (int r = 0; r < 2; ++r) {
		//		for (ydim = 10; ydim < 1000; ydim *= 10) {
		//			for (xdim = 10; xdim < 1000; xdim *= 10) {
		//				for (density = 0; density < 1; density += 0.2) {
		//					totalTime = 0;

		//					for (int e = 0; e < repeats; ++e) {
		//						sims[r]->clear();
		//						initialiseFrame(*sims[r], density);
		//						totalTime += sims[r]->stepForward(nFrames);
		//					}
		//					// clear up the space that the simulation is taking up
		//					sims[r]->clear();
		//					meanTime = totalTime / repeats;
		//					std::cout << simNames[r] << ", " << meanTime << std::endl;
		//					log << simNames[r] << ", " << ruleSet << "," << ydim << "," << xdim << "," << density << "," << meanTime << "," << "-1" << "," << "-1" << "," << "-1" << "," << sizeof(T) << "," << "\n";
		//				}

		//			}
		//		}

		//	}
		//}
		//catch (std::exception e) {
		//	error_log << "Sequential failure!: " << e.what() << "\n";
		//	// std::cout << "Got an error: " << e.what() << std::endl;
		//	//return 1;
		//}

		// CPU Parallelised

		//try {
		//	for (int r = 2; r < 6; ++r) {
		//		for (ydim = 10; ydim < 10000; ydim *= 10) {
		//			for (xdim = 10; xdim < 10000; xdim *= 10) {
		//				for (float density = 0; density < 1; density += 0.2) {
		//					for (nSegments = 1; nSegments <= std::thread::hardware_concurrency(); nSegments*=2)
		//					{
		//						totalTime = 0;
		//						params[0] = nSegments;
		//						params[1] = ydim;
		//						params[2] = xdim;
		//						sims[r]->setParams(params);
		//						for (int e = 0; e < repeats; ++e) {
		//							sims[r]->clear();
		//							initialiseFrame(*sims[r], density);
		//							std::cout << "Running CPU with ydim " << sims[r]->getYDim() << " and xdim " << sims[r]->getXDim() << std::endl;
		//							totalTime += sims[r]->stepForward(nFrames);
		//						}
		//						// clear up the space that the simulation is taking up
		//						sims[r]->clear();
		//						meanTime = totalTime / repeats;
		//						std::cout << simNames[r] << ", " << meanTime << std::endl;
		//						log << simNames[r] << ", " << ruleSet << "," << ydim << "," << xdim << "," << density << "," << meanTime << "," << nBlocks << "," << nThreads << "," << nSegments << "," << sizeof(T) << "," << "\n";
		//					}
		//				}
		//			}
		//		}

		//	}
		//}
		//catch (std::exception e) {
		//	error_log << "CPU failure!: " << e.what() << "\n";
		//	//std::cout << "Got an error: " << e.what() << std::endl;
		//	//return 1;
		//}

		// GPU Parallelised
		try {
			int runs = 0;
			for (int r = 6; r < 10; ++r) {
				for (ydim = 10; ydim < 1000; ydim *= 10) {
					for (xdim = 10; xdim < 1000; xdim *= 10) {
						for (float density = 0.05; density < 1; density += 1) {

							for (nBlocks = 64; nBlocks < 128; nBlocks*=2)
							{
								for (nThreads = 32; nThreads < 64; nThreads*=2)
								{
									for (nSegments = 8; nSegments < 16; nSegments *= 8)
									{
										try {
											++runs;
											std::cout << "Run: " << runs << std::endl;
											std::cout << "Using " << ruleSet << " with density " << density << std::endl;
											totalTime = 0;
											// params are persistent across runs
											params[0] = nSegments;
											params[1] = ydim;
											params[2] = xdim;
											params[3] = nBlocks;
											params[4] = nThreads;
											sims[r]->setParams(params);
											for (int e = 0; e < repeats; ++e) {
												sims[r]->clear();
												initialiseFrame(*sims[r], density);
												totalTime += sims[r]->stepForward(nFrames);
											}
											// clear up the space that the simulation is taking up
											sims[r]->clear();
											meanTime = totalTime / repeats;
											std::cout << simNames[r] << ", " << meanTime << std::endl;
											log << simNames[r] << ", " << ruleSet << "," << ydim << "," << xdim << "," << density << "," << meanTime << "," << nBlocks << "," << nThreads << "," << nSegments << "," << sizeof(T) << "," << "\n";
										}
										catch(std::exception f)
										{
											error_log << "Failed on blocks: " << nBlocks << ", threads: " << nThreads << ", and segments: " << nSegments << "\n";
											std::cout << "Failed on blocks: " << nBlocks << ", threads: " << nThreads << ", and segments: " << nSegments << "\n";
											error_log << "Exception was: " << f.what() << std::endl;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		catch (std::exception e) {
			error_log << "GPU failure!: " << e.what() << "\n";
			//std::cout << "Got an error: " << e.what() << std::endl;
			//return 1;
		}
	}
	return true;
}


int main() {
	Stopwatch timer;
	timer.reset();
	int ydim = 200, xdim = 200, nFrames = 1;
	std::string ID;
	std::cout << "Please enter the Computer ID (single digit): " << std::endl;
	std::cin >> ID;
	std::ofstream log{ "results_" + ID + ".out" };
	std::ofstream error_log{ "error_log.txt" };
	log << "Simulator,Ruleset,YDimension,XDimension,Density,MeanTime,nBlocks,nThreads,nSegments,storageSize(bytes)\n";

	runSimulations<int>(log, error_log, ydim, xdim, nFrames);
	runSimulations<long long int>(log, error_log, ydim, xdim, nFrames);

	log.close();
	error_log.close();
	std::cout << "Finished testing" << std::endl;
	std::cout << "Testing took " << timer.elapsed() << " seconds" << std::endl;
	getchar();
	return 0;
}
