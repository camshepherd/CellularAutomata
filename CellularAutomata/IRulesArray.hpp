#pragma once
#include <vector>
namespace CellularAutomata {
	/** Interface defining necessary functionality for any model definition
	*/
	template <typename T>
	class IRulesArray {
	protected:
		int y_dim = 1, x_dim = 1;
	public:
		int l = 2;
		/** Get the highest state that can represent a valid state
		*/
		virtual T getMaxValidState() const = 0;

		/** Whether the given state is valid in the context of the model
		@param cellState: The cell state representation value to be evaluated
		*/
		virtual bool isValid(T cellState) const = 0;

		/** Calculate the subsequent state of the targeted cell in the given frame
		@param cells: The complete frame containing the target cell
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#endif
#ifndef __CUDACC__
#define CUDA_FUNCTION 
#endif
		CUDA_FUNCTION virtual T getNextState(T* cells, int y, int x) const = 0;

		/** Set the frame dimensions to be used
		*/
		virtual bool setFrameDimensions(int y, int x){
			y_dim = y;
			x_dim = x;
			return true;
		}
	};
}