#pragma once
#include "IRules.h"

// Basic class to simulate BL model type I; cars go negative to positive on both axes.
class RulesBML :
	public IRules
{
public:
	RulesBML();
	~RulesBML();

	virtual bool isValid(int cellState) const override;
	virtual int getNextState(const std::vector<std::vector<int>>& cells, int y, int x) const override;
};

