#pragma once

#include "SimulatorGPU.hpp"
#include "IDeadZoneHandlerArray.hpp"
#include "ISegmenter.hpp"

namespace CellularAutomata {
	/** Extension of SimulatorGPU to keep track of which parts of the frame could change in each timestep, avoiding computations on those that are static
	*/
	template <typename T>
	class SimulatorGPUZoning : public SimulatorGPU<T> {
	protected:
	public:
		/** Constructor 1. Create the simulator utilising the given dependencies
		*/
		SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter);

		/** Constructor 2. Create the simulator utilising the given dependencies and defining the number of blocks and threads to use for simulation
		 */
		SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads);
		
		/** Destructor 1. Default destructor
		*/
		~SimulatorGPUZoning();

		/** Simulate subsequent frames of the simulation. Default is 1
		*/
		virtual double stepForward(int steps = 1) override;

		virtual bool setParams(int* list) override;
	};
}
#include "SimulatorGPUZoning.inl"