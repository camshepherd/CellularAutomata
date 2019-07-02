#pragma once
#include "SimulatorVector.h"

#include "ISegmenter.h"
#include <thread>
#include <vector>

// ISimulator implementation to allow for the utilisation of multiple CPU threads
class SimulatorCPU :
	public SimulatorVector
{
protected:

	const ISegmenter& segmenter;
public:
	SimulatorCPU(int ydim, int xdim, IRules& rules, ISegmenter& segmenter);
	~SimulatorCPU();

	virtual bool stepForwardRegion(int y_min, int y_max, int x_min, int x_max);
	virtual double stepForward(int steps = 1) override;
	virtual int getMaxValidState() override;
};
