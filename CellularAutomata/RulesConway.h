#pragma once
#include "IRules.h"

// Rules class to implement Conway's Game of Life
// Assumes a data representation where 0 is dead, and 1 is alive
class RulesConway :
	public IRules
{
private:
	const int live_min, live_max, birth_min, birth_max;
	const int cell_min, cell_max;
	int countNeighours(const std::vector<std::vector<int>>& cells, int y, int x) const;
public:
	RulesConway();
	RulesConway(int _live_min, int _live_max, int _birth_min, int _birth_max, int _cell_min, int _cell_max);
	~RulesConway();
	virtual bool isValid(int cellState) const override;
	virtual int getNextState(const std::vector<std::vector<int>>& cells, int y, int x) const override;
};

