#pragma once
#include "ISimulator.h"
#include "IRules.h"
#include <thread>
#include <vector>

// Simulator to utilise the full power of multi-core CPUs
class SimulatorCPU :
	public ISimulator
{
private:
	std::vector<std::vector<std::vector<int>>> cellStore;
	const int y_dim, x_dim;

	const IRules& rules;
	
public:
	SimulatorCPU(int ydim, int xdim, IRules& rules);
	~SimulatorCPU();
	virtual bool clear(bool addBlankFirstFrame = true) override;
	virtual int getNumFrames() const override;

	virtual bool setCell(int y, int x, int new_val, int t = -1) override;
	virtual bool blankFrame() override;

	virtual int getCell(int y, int x, int t = -1) const override;

	bool stepForwardRegion(int y_min, int y_max, int x_min, int x_max);
	bool stepForward(int steps = 1) override;
	bool stepForwardTime(double seconds) override;
};

