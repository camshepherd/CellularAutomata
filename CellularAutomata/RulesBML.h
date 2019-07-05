#pragma once
#include "IRules.h"

/** A model to simulate the Biham-Middleton-Levine Trafic model. The model is hard-coded and parameters of it cannot be altered
*/
template <typename T>
class RulesBML :
	public IRules<T>
{
public:
	/** Constructor 1. Default Constructor
	*/
	RulesBML();

	/** Destructor 1. Default destructor
	*/
	~RulesBML();

	/** Whether the given state is valid or not
	@param cellState: The cell state representation to be evaluated
	*/
	virtual bool isValid(T cellState) const override;

	/** Calculate the next state of the specified cell
	@param cells: The frame in which the cell is to be evaluated
	@param y: The y-coordinate of the target cell
	@param x: The x-coordinate of the target cell
	*/
	virtual T getNextState(const std::vector<std::vector<T>>& cells, int y, int x) const override;

	/** Get the maximum value that can be used to represent a cell's state
	*/
	virtual T getMaxValidState() const override;
};

#include "RulesBML.inl"