#pragma once
#include "SimulatorSequential.h"
#include "IDeadZoneHandler.h"
class SimulatorSequentialZoning :
	public SimulatorSequential
{

protected:
	IDeadZoneHandler& zoner;
public:
	SimulatorSequentialZoning(int y, int x, IRules& rules, IDeadZoneHandler& zoner);
	~SimulatorSequentialZoning();
	double stepForward(int steps = 1) override;
};

