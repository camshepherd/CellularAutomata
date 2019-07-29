#pragma once
#include "SimulatorVector.hpp"
#include "IRules.hpp"
#include <vector>

namespace CellularAutomata {
	/** Simulate synchronous Cellular Automata sequenetially in a single CPU thread
	*/
	template <typename T>
	class SimulatorSequential :
		public SimulatorVector<T>
	{
	public:
		/** Constructor 1. Create the simulator with the given dependency
		@param y: The size of the simulation in the y-directoin
		@param x: The size of the simulation in the x-direction
		*/
		SimulatorSequential(const int y, const int x, const IRules<T>& _rules);

		/** Destructor 1. Default destructor
		*/
		~SimulatorSequential();

		/** Simulate subsequent steps in the simulation
		@param steps: Number of steps to simulate by. Default is 1
		*/
		virtual double stepForward(int steps = 1) override;

		/** Get the maximum value that can be used to represent a state in the model
		*/
		virtual T getMaxValidState() override;

		virtual bool setParams(int* list) override;
	};
}
#include "SimulatorSequential.inl"