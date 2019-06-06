#pragma once
#include "ISimulator.h"
#include <vector>

class SimulatorConway :
	public ISimulator
{
private:
	std::vector<std::vector<std::vector<int>>> cellStore;
	int y_dim, x_dim;
	static const int cell_min = 0;
	static const int cell_max = 3;

	static const int live_min = 2;
	static const int live_max = 3;
	static const int birth_min = 3;
	static const int birth_max = 3;

	int updateCell(int y, int x, int t =-1);
	int countNeighbours(int y, int x, int t=-1);

public:
	SimulatorConway(int y, int x);
	~SimulatorConway();
	virtual bool clear(bool addBlankFirstFrame = true);
	virtual int getNumFrames();

	virtual bool setCell(int y, int x, int new_val, int t = -1);
	virtual bool blankFrame();

	virtual int getCell(int y, int x, int t = -1);

	bool stepForward(int steps = 1);
};

