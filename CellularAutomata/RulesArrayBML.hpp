#pragma once
#include "IRulesArray.hpp"
#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#endif
#ifndef __CUDACC__
#define CUDA_FUNCTION 
#endif


namespace CellularAutomata {
	/** A model to simulate the Biham-Middleton-Levine Traffic model. The model is hard-coded and parameters of it cannot be altered
	*/
	template <typename T>
	class RulesArrayBML :
		public IRulesArray<T>
	{
	public:
		/** Constructor 1. Default Constructor
		*/
		CUDA_FUNCTION RulesArrayBML();

		/** Constructor 2. Specify the dimensions of the frames to be used
		*/
		CUDA_FUNCTION RulesArrayBML(int y_dim, int x_dim);

		/** Destructor 1. Default destructor
		*/
		CUDA_FUNCTION ~RulesArrayBML();

		/** Whether the given state is valid or not
		@param cellState: The cell state representation to be evaluated
		*/
		CUDA_FUNCTION virtual bool isValid(T cellState) const override;

		/** Calculate the next state of the specified cell
		@param cells: The frame in which the cell is to be evaluated
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		CUDA_FUNCTION virtual T getNextState(T* cells, int y, int x) const override;

		/** Get the maximum value that can be used to represent a cell's state
		*/
		CUDA_FUNCTION virtual T getMaxValidState() const override;
		
	};
}
#include "RulesArrayBML.inl"