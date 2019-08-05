#pragma once

#include "SimulatorCPU.hpp"
#include "IDeadZoneHandler.hpp"
#include "ISegmenter.hpp"

namespace CellularAutomata {
	/** Extension of SimulatorCPU to keep track of which parts of the frame could change in each timestep, avoiding computations on those that are static
	*/
	template <typename T>
	class SimulatorCPUZoning : public SimulatorCPU<T> {
	protected:
		IDeadZoneHandler<T>& zoner;
		
	public:
		/** Constructor 1. Create the simulator utilising the given dependencies
		*/
		SimulatorCPUZoning(int y, int x, IRules<T>& rules, ISegmenter& segmenter, IDeadZoneHandler<T>& zoner);

		/** Destructor 1. Default destructor
		*/
		~SimulatorCPUZoning();

		/** Simulate subsequent frames of the simulation. Default is 1
		*/
		virtual double stepForward(int steps = 1) override;

		/** Simulate the next state of cells in the given region
		@param y_min: The lowest y-coordinate of the region
		@param y_max: The largest y-coordinate of the region
		@param x_min: The lowest x-coordinate of the region
		@param x_max: The largest x-coordinate of the region
		*/
		virtual bool stepForwardRegion(int y_min, int y_max, int x_min, int x_max) override;

		/** Set the dimensions of the frame that the zoner is handling
		@param y: The size of the frame in the y dimension
		@param x: The size of the frame in the x dimension
		*/
		virtual bool setDimensions(int y, int x) override;
	};
}
#include "SimulatorCPUZoning.inl"