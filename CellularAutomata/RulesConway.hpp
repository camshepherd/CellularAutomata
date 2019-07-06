#pragma once
#include "IRules.hpp"

namespace CellularAutomata {
	/** A model implementing the rules of Conway's Game of Life
	*/
	template <typename T>
	class RulesConway :
		public IRules<T>
	{
	protected:
		const int live_min, live_max, birth_min, birth_max;
		const int cell_min, cell_max;

		/** Count the number of neighbours that the targeted cell has
		@param cells: The frame to be used
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		int countNeighours(const std::vector<std::vector<T>>& cells, int y, int x) const;
	public:
		/** Constructor 1. Default constructor - uses Conway's own parameters for the model
		*/
		RulesConway();

		/** Constructor 2. Explicit declarations of the states of the birth and death values
		@param live_min: The minimum number of neighbours with which a cell will remain alive
		@param live_max: The maximum number of neighbours with which a cell will remain alive
		@param birth_min: The minimum number of neighbours with which a dead cell will be born
		@param birth_max: The maximum number of neighbours with which a dead cell will be born
		*/
		RulesConway(int live_min, int live_max, int birth_min, int birth_max, int cell_min, int cell_max);

		/** Destructor 1. Default destructor
		*/
		~RulesConway();

		/** Whether the given cell state is valid in the model
		@param cellState: The cell state to be evaluated
		*/
		virtual bool isValid(T cellState) const override;

		/** Calculate the next state for the specified cell
		@param cells: The frame in which the cell is to be evaluated
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		virtual T getNextState(const std::vector<std::vector<T>>& cells, int y, int x) const override;

		/** Get the maximum value that can be used to represent a state in the model
		*/
		virtual T getMaxValidState() const override;
	};
}
#include "RulesConway.inl"