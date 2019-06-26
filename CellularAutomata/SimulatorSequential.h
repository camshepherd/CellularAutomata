#pragma once
#include "ISimulator.h"
#include "IRules.h"
#include <vector>

// ISimulator implementation to simulate CAs sequentially - in a single thread
class SimulatorSequential :
	public ISimulator
{
protected:
	std::vector<std::vector<std::vector<int>>> cellStore;
	const int y_dim, x_dim;

	const IRules& rules;
public:
	SimulatorSequential(const int y, const int x, const IRules& _rules);
	~SimulatorSequential();
	virtual bool clear(bool addBlankFirstFrame = true) override;
	virtual int getNumFrames() const override;

	virtual bool setCell(int y, int x, int new_val, int t = -1) override;
	virtual bool blankFrame() override;

	virtual int getCell(int y, int x, int t = -1) const override;

	double stepForward(int steps = 1) override;
	int getYDim() override;
	int getXDim() override;
	bool copyFrame() override;
};

