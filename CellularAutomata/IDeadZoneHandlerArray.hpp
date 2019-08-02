#pragma once

#include <tuple>
#include <vector>

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#endif
#ifndef __CUDACC__
#define CUDA_FUNCTION 
#endif

namespace CellularAutomata {
	/** Interface definition for classes that keep track of whether cells
		have the potential to change on each step.
		*/
	template <typename T>
	class IDeadZoneHandlerArray
	{
	public:
		/** Constructor 1. Default constructor
		*/
		CUDA_FUNCTION IDeadZoneHandlerArray() {};

		/** Destructor 1. Default destructor
		*/
		CUDA_FUNCTION ~IDeadZoneHandlerArray() {};

		/** Update the internal state of regions of activity
		@param frame1: The first frame; to be compared against frame2
		@param frame2: The second frame; to be compared against frame1
		*/
		CUDA_FUNCTION bool virtual updateDeadZones(T* frame1, T* frame2) = 0;

		/** Whether the targeted cell has the potential to change state in the next frame
		@param y: The cell's y-coordinate
		@param x: The cell's x-coordinate
		*/
		CUDA_FUNCTION bool virtual isLive(int y, int x) = 0;


		CUDA_FUNCTION bool virtual setDimensions(int* dims) = 0;

		CUDA_FUNCTION bool virtual refresh() = 0;
	};
}