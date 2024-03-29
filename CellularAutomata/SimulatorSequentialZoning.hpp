#pragma once
#include "SimulatorSequential.hpp"
#include "IDeadZoneHandler.hpp"

namespace CellularAutomata {
	/** Extension of SimulatorSequential that keeps track of which regions have the potential for change and hence require simulation
	*/
	template <typename T>
	class SimulatorSequentialZoning :
		public SimulatorSequential<T>
	{

	protected:
		IDeadZoneHandler<T>& zoner;
	public:
		/** Constructor 1. Construct the simulator from the injected dependencies
		*/
		SimulatorSequentialZoning(int y, int x, IRules<T>& rules, IDeadZoneHandler<T>& zoner);

		/** Destructor 1. Default destructor
		*/
		~SimulatorSequentialZoning();

		/** Simulate forward the specified number of timesteps.
		@param steps: Number of timesteps to simulate forward by. Default is 1
		*/
		double stepForward(int steps = 1) override;

		/** Set the dimensions of the frame that the zoner is handling
		@param y: The size of the frame in the y dimension
		@param x: The size of the frame in the x dimension
		*/
		virtual bool setDimensions(int y, int x) override;
	};
}
#include "SimulatorSequentialZoning.inl"