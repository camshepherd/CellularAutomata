#pragma once
#include <vector>
// Interface defining functionality necessary for any generic 
// set of automaton rules

class IRules {
public:
	virtual bool isValid(int cellState) = 0;
	virtual int getNextState(std::vector<std::vector<int>>& cells, int y, int x) = 0;
};