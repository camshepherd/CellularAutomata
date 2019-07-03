#pragma once
#include "SimulatorVector.h"
#include "IRules.h"
#include <vector>

/** Simulate synchronous Cellular Automata sequenetially in a single CPU thread
*/
class SimulatorSequential :
	public SimulatorVector
{
public:
	/** Constructor 1. Create the simulator with the given dependency
	@param y: The size of the simulation in the y-directoin
	@param x: The size of the simulation in the x-direction
	*/
	SimulatorSequential(const int y, const int x, const IRules& _rules);

	/** Destructor 1. Default destructor
	*/
	~SimulatorSequential();

	/** Simulate subsequent steps in the simulation
	@param steps: Number of steps to simulate by. Default is 1
	*/
	virtual double stepForward(int steps = 1) override;

	/** Get the maximum value that can be used to represent a state in the model
	*/
	virtual int getMaxValidState() override;
};

