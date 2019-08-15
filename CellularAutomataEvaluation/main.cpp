#include "pch.h"
#include <iostream>
#include <memory>
#include <map>
#include <fstream>
#include "RulesArrayConway.hpp"
#include "SimulatorGPU.hpp"
#include "RulesArrayBML.hpp"
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
bool runSimulations(std::ofstream& log, std::ofstream& error_log)
{
	Stopwatch timer{};
	timer.reset();
	error_log << "Just to prove that this actually works...\r\n";
	int sizeID = sizeof(T), numRepeats = 4, numFrames = 20;
	// Generate the sequential results for changing dimensions
	RulesBML<T> bml{};
	RulesConway<T> con{};
	float meanTime,density = 0.3;
	SegmenterStrips seg{ 0 };


	SimulatorSequential<T> seq{ 64,64,bml };
	for(int y = 64; y <= 4096;y*=2)
	{
			meanTime = 0;
			seq.setDimensions(y, y);
			for(int r = 0; r < numRepeats; ++r)
			{
				seq.rebuildCellStore();
				initialiseFrame(seq, density);
				meanTime += seq.stepForward(numFrames);
			}
			meanTime /= numRepeats;
			log << "seq,bml," << y << "," << y << "," << density << "," << meanTime << ",-1,-1," << sizeID << std::endl;
		
	}
	std::cout << "done seq bml" << std::endl;
	seq.clear(false);
	SimulatorSequential<T>seq2{ 64,64,con };
	for (int y = 64; y <= 4096; y*=2)
	{
			meanTime = 0;
			seq2.setDimensions(y, y);
			for (int r = 0; r < numRepeats; ++r)
			{
				seq2.rebuildCellStore();
				initialiseFrame(seq2, density);
				meanTime += seq2.stepForward(numFrames);
			}
			meanTime /= numRepeats;
			log << "seq,con," << y << "," << y << "," << density << "," << meanTime << ",-1,-1," << sizeID << std::endl;

	}
	seq2.clear(false);
	std::cout << "Done seq conway" << std::endl;
	ZonerPixels<T> zoner{64,64};
	SimulatorSequentialZoning<T> seqzon{ 64,64,bml,zoner };
	for (int y = 64; y <= 4096; y *= 2)
	{
			meanTime = 0;
			seqzon.setDimensions(y, y);
			zoner.setDimensions(y, y);
			for (int r = 0; r < numRepeats; ++r)
			{
				seqzon.rebuildCellStore();
				initialiseFrame(seqzon, density);
				meanTime += seqzon.stepForward(numFrames);
			}
			meanTime /= numRepeats;
			log << "seqzon,bml," << y << "," << y << "," << density << "," << meanTime << ",-1,-1," << sizeID << std::endl;
	}
	seqzon.clear(false);
	std::cout << "done seq zoning bml" << std::endl;
	SimulatorSequentialZoning<T> seqzon2{ 64,64,con,zoner };
	for (int y = 64; y <= 4096; y *= 2)
	{
			meanTime = 0;
			seqzon2.setDimensions(y, y);
			for (int r = 0; r < numRepeats; ++r)
			{
				seqzon2.rebuildCellStore();
				initialiseFrame(seqzon2, density);
				meanTime += seqzon2.stepForward(numFrames);
			}
			meanTime /= numRepeats;
			log << "seqzon,con," << y << "," << y << "," << density << "," << meanTime << ",-1,-1," << sizeID << std::endl;
	}
	seqzon2.clear(false);
	
	std::cout << "done seqzon conway" << std::endl;
	int numThreads = std::thread::hardware_concurrency();
	
	

	SimulatorCPU<T> cpu{ 64,64,bml,seg };
	for (int y = 64; y <= 4096; y *= 2)
	{
			meanTime = 0;
			cpu.setDimensions(y, y);
			for (int r = 0; r < numRepeats; ++r)
			{
				cpu.rebuildCellStore();
				initialiseFrame(cpu, density);
				cpu.setLaunchParams(numThreads);
				meanTime += cpu.stepForward(numFrames);
			}
			meanTime /= numRepeats;
			log << "cpuhor,bml," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
		
	}
	std::cout << "done cpu bml horizontal";
	cpu.clear(false);

	SimulatorCPU<T> cpu2{ 64,64,con,seg };
	for (int y = 64; y <= 4096; y *= 2)
	{
			meanTime = 0;
			cpu2.setDimensions(y, y);
			for (int r = 0; r < numRepeats; ++r)
			{
				cpu2.rebuildCellStore();
				initialiseFrame(cpu2, density);
				cpu2.setLaunchParams(numThreads);
				meanTime += cpu2.stepForward(numFrames);
			}
			meanTime /= numRepeats;
			log << "cpuhor,con," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
		
	}
	std::cout << "Done cpu conway horizontal";




		seg.setOrientation(1);
	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		cpu.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			cpu.rebuildCellStore();
			initialiseFrame(cpu, density);
			cpu.setLaunchParams(numThreads);
			meanTime += cpu.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "cpuver,bml," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
	}
	std::cout << "done cpu bml vertical";
	cpu.clear(false);

	
	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		cpu2.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			cpu2.rebuildCellStore();
			initialiseFrame(cpu2, density);
			cpu2.setLaunchParams(numThreads);
			meanTime += cpu2.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "cpuver,con," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done cpu conway vertical";

	cpu2.clear(false);

	seg.setOrientation(0);


	SimulatorCPUZoning<T> cpuzon{ 64,64,con,seg,zoner };
	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		cpuzon.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			cpuzon.rebuildCellStore();
			initialiseFrame(cpuzon, density);
			cpuzon.setLaunchParams(numThreads);
			meanTime += cpuzon.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "cpuzonhor,con," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done cpu conway zoning horizontal";
	cpuzon.clear(false);


	SimulatorCPUZoning<T> cpuzon2{ 64,64,bml,seg,zoner };
	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		cpuzon2.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			cpuzon2.rebuildCellStore();
			initialiseFrame(cpuzon2, density);
			cpuzon2.setLaunchParams(numThreads);
			meanTime += cpuzon2.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "cpuzonhor,bml," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done cpu bml zoning horizontal";
	cpuzon2.clear(false);

	seg.setOrientation(0);


	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		cpuzon.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			cpuzon.rebuildCellStore();
			initialiseFrame(cpuzon, density);
			cpuzon.setLaunchParams(numThreads);
			meanTime += cpuzon.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "cpuzonhor,con," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done cpu conway zoning horizontal";
	cpuzon.clear(false);


	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		cpuzon2.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			cpuzon2.rebuildCellStore();
			initialiseFrame(cpuzon2, density);
			cpuzon2.setLaunchParams(numThreads);
			meanTime += cpuzon2.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "cpuzonhor,bml," << y << "," << y << "," << density << "," << meanTime << ",-1," << numThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done cpu bml zoning horizontal";
	cpuzon2.clear(false);

	RulesArrayConway<T> conA{};
	RulesArrayBML<T> bmlA{};
	int gpuBlocks = 32, gpuThreads = 128;


	seg.setOrientation(0);
	
	SimulatorGPU<T> gpuhor{ 64,64,conA,seg,32, 128 };
	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		gpuhor.setLaunchParams(gpuBlocks, gpuThreads, gpuBlocks * gpuThreads);
		gpuhor.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			gpuhor.clear();
			//gpuhor.rebuildCellStore();
			initialiseFrame(gpuhor, density);
			meanTime += gpuhor.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "gpuhor,con," << y << "," << y << "," << density << "," << meanTime << "," << gpuBlocks << "," << gpuThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done gpu conway horizontal";
	gpuhor.clear(false);




	SimulatorGPU<T> gpuhor2{ 64,64,bmlA,seg,32, 128};
	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		gpuhor2.setLaunchParams(gpuBlocks, gpuThreads, gpuBlocks * gpuThreads);
		gpuhor2.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			gpuhor2.clear();
			//gpuhor.rebuildCellStore();
			initialiseFrame(gpuhor2, density);
			meanTime += gpuhor2.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "gpuhor,bml," << y << "," << y << "," << density << "," << meanTime << "," << gpuBlocks << "," << gpuThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done gpu bml horizontal";
	gpuhor.clear(false);








	seg.setOrientation(1);

	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		gpuhor.setLaunchParams(gpuBlocks, gpuThreads, gpuBlocks * gpuThreads);
		gpuhor.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			gpuhor.clear();
			//gpuhor.rebuildCellStore();
			initialiseFrame(gpuhor, density);
			meanTime += gpuhor.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "gpuver,con," << y << "," << y << "," << density << "," << meanTime << "," << gpuBlocks << "," << gpuThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done gpu conway horizontal";
	gpuhor.clear(false);



	for (int y = 64; y <= 4096; y *= 2)
	{
		meanTime = 0;
		gpuhor2.setLaunchParams(gpuBlocks, gpuThreads, gpuBlocks * gpuThreads);
		gpuhor2.setDimensions(y, y);
		for (int r = 0; r < numRepeats; ++r)
		{
			gpuhor2.clear();
			//gpuhor.rebuildCellStore();
			initialiseFrame(gpuhor2, density);
			meanTime += gpuhor2.stepForward(numFrames);
		}
		meanTime /= numRepeats;
		log << "gpuver,bml," << y << "," << y << "," << density << "," << meanTime << "," << gpuBlocks << "," << gpuThreads << "," << sizeID << std::endl;
	}
	std::cout << "Done gpu bml horizontal";
	gpuhor.clear(false);


	return true;
}


int main() {
	Stopwatch timer;
	timer.reset();
	int ID;
	std::cout << "Please enter the Computer ID (single digit): " << std::endl;
	std::cin >> ID;
	std::ofstream log{ "results_" + std::to_string(ID) + ".out" };
	std::ofstream error_log{ "error_log.txt" };
	log << "Simulator,Ruleset,YDimension,XDimension,Density,MeanTime,nBlocks,nThreads,storageSize(bytes)\n";

	runSimulations<bool>(log, error_log);
	runSimulations<short int>(log, error_log);
	runSimulations<int>(log, error_log);
	// No point doing long as it's the same as int
	runSimulations<long long int>(log, error_log);

	log.close();
	error_log.close();
	std::cout << "Finished testing" << std::endl;
	std::cout << "Testing took " << timer.elapsed() << " seconds" << std::endl;
	getchar();
	return 0;
}
