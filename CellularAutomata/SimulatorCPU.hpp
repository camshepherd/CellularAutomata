#pragma once
#include "SimulatorVector.hpp"

#include "ISegmenter.hpp"
#include <thread>
#include <vector>

namespace CellularAutomata {
	/** Simulator to simulate the given model while making full use of multiple threads on the CPU
	*/
	template <typename T>
	class SimulatorCPU :
		public SimulatorVector<T>
	{
	protected:
		const ISegmenter& segmenter;
		int nSegments;
	public:
		/** Constructor 1. Create the simulator using the injected dependencies
		@param ydim: The size of the simulation in the y-axis
		@param xdim: The size of the simulation in the x-axis
		@param rules: The rules to use to simulate the model
		@param segmenter: The class to use to segment up the frame to the individual CPU threads
		*/
		SimulatorCPU(int ydim, int xdim, IRules<T>& rules, ISegmenter& segmenter);

		/** Destructor 1. Default destructor
		*/
		~SimulatorCPU();

		/** Calculate the next state of all cells in the given region
		@param y_min: The lowest y-coordinate of the region
		@param y_max: The largest y-coordinate of the region
		@param x_min: The lowest x-coordinate of the region
		@param x_max: The largest x-coordinate of the region
		*/
		virtual bool stepForwardRegion(int y_min, int y_max, int x_min, int x_max);

		/** Simulate additional frames of the simulation, utilising all threads on the CPU
		@param steps: The number of steps to move the simulation forward by. Default is 1
		*/
		virtual double stepForward(int steps = 1) override;

		/** Get the maximum valid value to represent states in the model
		*/
		virtual T getMaxValidState() override;

		virtual bool setParams(int* list) override;
	};
}

#include "SimulatorCPU.inl"