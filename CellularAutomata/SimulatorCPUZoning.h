#pragma once

#include "SimulatorCPU.h"
#include "IDeadZoneHandler.h"
#include "ISegmenter.h"

class SimulatorCPUZoning : public SimulatorCPU {
protected:
	IDeadZoneHandler& zoner;
public:
	SimulatorCPUZoning(int y, int x, IRules& rules, ISegmenter& segmenter, IDeadZoneHandler& zoner);
	~SimulatorCPUZoning();
	virtual double stepForward(int steps = 1) override;
	virtual bool stepForwardRegion(int y_min, int y_max, int x_min, int x_max) override;
};