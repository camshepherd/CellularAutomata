#pragma once
#include "SimulatorVector.h"
#include "IRules.h"
#include <vector>

// ISimulator implementation to simulate CAs sequentially - in a single thread
class SimulatorSequential :
	public SimulatorVector
{
public:
	SimulatorSequential(const int y, const int x, const IRules& _rules);
	~SimulatorSequential();
	virtual double stepForward(int steps = 1) override;
	virtual int getMaxValidState() override;
};

