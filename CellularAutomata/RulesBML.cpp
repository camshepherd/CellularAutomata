#include "RulesBML.h"



RulesBML::RulesBML()
{
	// doesn't need to do anything as there is no customisation of this simulation is static.
}


RulesBML::~RulesBML()
{
}

bool RulesBML::isValid(int cellState) const {
	switch (cellState) {
	case 0:
		return true;
		break;
	case 1:
		return true;
		break;
	case 2:
		return true;
		break;
	default:
		return false;
		break;
	}
}

int RulesBML::getNextState(const std::vector<std::vector<int>>& cells, int y, int x) const {
	int y_dim = cells.size();
	int x_dim = cells[0].size();
	switch (cells[y][x]) {
	case 0:
		if (cells[y][(x - 1 + x_dim) % x_dim] == 1) {
			return 1;
		}
		else if (cells[(y - 1 + y_dim) % y_dim][x] == 2) {
			return 2;
		}
		else {
			return 0;
		}
		break;
	case 1:
		if (cells[y][(x + 1) % x_dim] == 0) {
			if (cells[(y - 1 + y_dim) % y_dim ][x] == 2) {
				return 2;
			}
			else {
				return 0;
			}
		}
		else {
			return 1;
		}
		break;
	case 2:
		if (cells[(y + 1) % y_dim + y_dim][x] == 0) {
			return 0;
		}
		else {
			return 0;
		}
		break;
	default:
		throw std::runtime_error("An invalid state got into the cell store");
	};
}