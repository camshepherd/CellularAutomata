#include "RulesConway.h"



RulesConway::RulesConway() : live_min(2), live_max(3), birth_min(3), birth_max(3), cell_min(0), cell_max(1)
{
}

RulesConway::RulesConway(int _live_min, int _live_max, int _birth_min, int _birth_max, int _cell_min, int _cell_max) : live_min(_live_min), live_max(_live_max), birth_min(_birth_min), birth_max(_birth_max), cell_min(_cell_min), cell_max(_cell_max)
{
}


RulesConway::~RulesConway()
{
}



bool RulesConway::isValid(int cellState) const {
	if (cellState >= cell_min && cellState <= cell_max) {
		return true;
	}
	else {
		return false;
	}
}

int RulesConway::getNextState(const std::vector<std::vector<int>>& cells, int y, int x) const {
	int count = countNeighours(cells, y, x);
	if (cells[y][x]) {
		//alive
		if (count >= live_min && count <= live_max) {
			return 1;
		}
	}
	else {
		//dead
		if (count >= birth_min && count <= birth_max) {
			return 1;
		}
	}
	return 0;
}

int RulesConway::countNeighours(const std::vector<std::vector<int>>& cells, int y, int x) const {
	int count = 0;
	// assumed that the world will be a rectangle
	int y_dim = cells.size();
	int x_dim = cells[0].size();
	for (int _y = y - 1; _y <= y + 1; ++_y) {
		for (int _x = x - 1; _x <= x + 1; ++_x) {
			if (_y == y && _x == x) {
				continue;
			}
			else if (cells[(_y + y_dim)%y_dim][(_x+x_dim)%x_dim]) {
				count += 1;
			}
		}
	}
	return count;
}