#pragma once
#include <vector>
namespace CellularAutomata {
	/** Interface defining necessary functionality for any model definition
	*/
	template <typename T>
	class IRulesArray {
	protected:
		int y_dim, x_dim;
	public:
		/** Get the highest state that can represent a valid state
		*/
		virtual T getMaxValidState() const = 0;

		/** Whether the given state is valid in the context of the model
		@param cellState: The cell state representation value to be evaluated
		*/
		virtual bool isValid(T cellState) const = 0;

		/** Calcualate the subsequent state of the targeted cell in the given frame
		@param cells: The complete frame containing the target cell
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		virtual T getNextState(T* cells, int y, int x) const = 0;

		/** Set the frame dimensions to be used
		*/
		virtual bool setFrameDimensions(int y, int x){
			y_dim = y;
			x_dim = x;
			return true;
		}
	};
}