#pragma once
#include "SimulatorSequential.h"
#include "IDeadZoneHandler.h"

/** Extension of SimulatorSequential that keeps track of which regions have the potential for change and hence require simulation
*/
class SimulatorSequentialZoning :
	public SimulatorSequential
{

protected:
	IDeadZoneHandler& zoner;
public:
	/** Constructor 1. Construct the simulator from the injected dependencies
	*/
	SimulatorSequentialZoning(int y, int x, IRules& rules, IDeadZoneHandler& zoner);

	/** Destructor 1. Default destructor
	*/
	~SimulatorSequentialZoning();

	/** Simulate forward the specified number of timesteps. 
	@param steps: Number of timesteps to simulate forward by. Default is 1
	*/
	double stepForward(int steps = 1) override;
};

