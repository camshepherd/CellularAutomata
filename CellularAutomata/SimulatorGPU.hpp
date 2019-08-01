#pragma once
#include "SimulatorArray.hpp"

#include "ISegmenter.hpp"
#include <vector>

namespace CellularAutomata {
	/** Simulator to simulate the given model while making full use of multiple threads on the GPU
	*/
	template <typename T>
	class SimulatorGPU :
		public SimulatorArray<T>
	{
	protected:
		const ISegmenter& segmenter;
		int* segments;
		int nBlocks, nThreads, nSegments;
		int maxY = 1000, maxX = 1000;
	public:
		/** Constructor 1. Create the simulator using the injected dependencies
		@param ydim: The size of the simulation in the y-axis
		@param xdim: The size of the simulation in the x-axis
		@param rules: The rules to use to simulate the model
		@param segmenter: The class to use to segment up the frame to the individual CPU threads
		*/
		SimulatorGPU(int ydim, int xdim, IRulesArray<T>& rules, ISegmenter& segmenter);

		/** Constructor 2. Create the simulator using the injected dependencies, and specify the numbers of blocks and threads to use
		@param ydim: The size of the simulation in the y-axis
		@param xdim: The size of the simulation in the x-axis
		@param rules: The rules to use to simulate the model
		@param segmenter: The class to use to segment up the frame to the individual CPU threads
		*/
		SimulatorGPU(int ydim, int xdim, IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads);

		/** Destructor 1. Default destructor
		*/
		~SimulatorGPU();

		/** Simulate additional frames of the simulation, utilising all threads on the CPU
		@param steps: The number of steps to move the simulation forward by. Default is 1
		*/
		virtual double stepForward(int steps = 1) override;

		/** Get the maximum valid value to represent states in the model
		*/
		virtual T getMaxValidState() override;

		/** Set the number of blocks and threads to use to run the CUDA kernel for stepping the simulation
		*/
		bool setLaunchParams(int nBlocks, int nThreads);

		virtual bool setParams(int* list) override;
	};
}

#include "SimulatorGPU.inl"