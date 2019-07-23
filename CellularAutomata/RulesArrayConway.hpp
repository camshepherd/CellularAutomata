#pragma once
#include "IRulesArray.hpp"

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#endif
#ifndef __CUDACC__
#define CUDA_FUNCTION 
#endif

namespace CellularAutomata {
	/** A model implementing the rules of Conway's Game of Life with
	an underlying data structure of a 1D C-style array
	*/
	template <typename T>
	class RulesArrayConway :
		public IRulesArray<T>
	{
	protected:
		const int live_min, live_max, birth_min, birth_max;
		const int cell_min, cell_max;
		/** Count the number of neighbours that the targeted cell has
		@param cells: The frame to be used
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		CUDA_FUNCTION int countNeighours(const T* cells, int y, int x) const;
	public:
		int k = 1;
		/** Constructor 1. Default constructor - uses Conway's own parameters for the model. Defaults to 3x3 frame
		*/
		CUDA_FUNCTION RulesArrayConway();

		/** Constructor 2. Only specify the size of the frames
		@param y_dim: The size of the frame in the y direction
		@param x_dim: The size of the frame in the x direction
		*/
		CUDA_FUNCTION RulesArrayConway(int y_dim, int x_dim);

		/** Constructor 3. Explicit declarations of the states of the birth and death values
		@param live_min: The minimum number of neighbours with which a cell will remain alive
		@param live_max: The maximum number of neighbours with which a cell will remain alive
		@param birth_min: The minimum number of neighbours with which a dead cell will be born
		@param birth_max: The maximum number of neighbours with which a dead cell will be born
		@param y_dim: The size of the frame in the y direction
		@param x_dim: The size of the frame in the x direction
		*/
		CUDA_FUNCTION RulesArrayConway(int live_min, int live_max, int birth_min, int birth_max, int cell_min, int cell_max, int y_dim, int x_dim);

		/** Destructor 1. Default destructor
		*/
		CUDA_FUNCTION ~RulesArrayConway();

		/** Whether the given cell state is valid in the model
		@param cellState: The cell state to be evaluated
		*/
		CUDA_FUNCTION virtual bool isValid(T cellState) const override;

		/** Calculate the next state for the specified cell
		@param cells: The frame in which the cell is to be evaluated
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		CUDA_FUNCTION virtual T getNextState(T* cells, int y, int x) const override;

		/** Get the maximum value that can be used to represent a state in the model
		*/
		CUDA_FUNCTION virtual T getMaxValidState() const override;

	};
}
#include "RulesArrayConway.inl"