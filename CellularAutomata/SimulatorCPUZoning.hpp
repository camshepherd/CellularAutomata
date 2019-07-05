#pragma once

#include "SimulatorCPU.h"
#include "IDeadZoneHandler.h"
#include "ISegmenter.h"

/** Extension of SimulatorCPU to keep track of which parts of the frame could change in each timestep, avoiding computations on those that are static
*/
class SimulatorCPUZoning : public SimulatorCPU {
protected:
	IDeadZoneHandler& zoner;
public:
	/** Constructor 1. Create the simulator utilising the given dependencies
	*/
	SimulatorCPUZoning(int y, int x, IRules& rules, ISegmenter& segmenter, IDeadZoneHandler& zoner);

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
};