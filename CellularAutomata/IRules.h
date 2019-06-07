#pragma once
#include <vector>
// Interface defining functionality necessary for any generic 
// set of automaton rules

class IRules {
public:
	virtual bool isValid(int cellState) const = 0;
	virtual int getNextState(const std::vector<std::vector<int>>& cells, int y, int x) const = 0;
};