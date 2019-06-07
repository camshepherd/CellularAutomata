#pragma once
#include "ISimulator.h"
#include "IRules.h"
#include <vector>

// ISimulator implementation to handle the simulation of Conway's Game of Life
class SimulatorConway :
	public ISimulator
{
private:
	std::vector<std::vector<std::vector<int>>> cellStore;
	const int y_dim, x_dim;

	const IRules& rules;

public:
	SimulatorConway(const int y, const int x, const IRules& _rules);
	~SimulatorConway();
	virtual bool clear(bool addBlankFirstFrame = true) override;
	virtual int getNumFrames() override;

	virtual bool setCell(int y, int x, int new_val, int t = -1) override;
	virtual bool blankFrame() override;

	virtual int getCell(int y, int x, int t = -1) override;

	bool stepForward(int steps = 1) override;
	bool stepForward(double seconds) override;
};

