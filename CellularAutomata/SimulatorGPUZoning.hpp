#pragma once

#include "SimulatorGPU.hpp"
#include "IDeadZoneHandlerArray.hpp"
#include "ISegmenter.hpp"
#include "ZonerArrayPixels.hpp"

namespace CellularAutomata {
	/** Extension of SimulatorGPU to keep track of which parts of the frame could change in each timestep, avoiding computations on those that are static
	*/
	template <typename T>
	class SimulatorGPUZoning : public SimulatorGPU<T> {
	protected:
		int *d_zoner_dims, *d_zoner_maxDims;
		ZonerArrayPixels<T>* d_zoner;
		int y_max, x_max;
		bool *d_zoner_a, *d_zoner_b;
	public:

		/** Constructor 2. Create the simulator utilising the given dependencies and defining the number of blocks and threads to use for simulation
		 */
		SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads, int y_max=10000, int x_max=10000);
		
		/** Destructor 1. Default destructor
		*/
		~SimulatorGPUZoning();

		/** Simulate subsequent frames of the simulation. Default is 1
		*/
		virtual double stepForward(int steps = 1) override;

		virtual bool setDimensions(int y, int x) override;
	};
}
#include "SimulatorGPUZoning.inl"