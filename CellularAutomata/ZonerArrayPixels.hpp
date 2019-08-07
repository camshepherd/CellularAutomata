#pragma once
#include "IDeadZoneHandlerArray.hpp"

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#endif
#ifndef __CUDACC__
#define CUDA_FUNCTION 
#endif

namespace CellularAutomata {
	/** Class to keep track of which pixels may change from frame to frame, and hence require simulation
	*/
	template <typename T>
	class ZonerArrayPixels :
		public IDeadZoneHandlerArray<T>
	{
	protected:
		bool *cellActivities, *rawActivities;
		int *dims, *maxDims;
	public:
		
		/** Constructor 1. Create a zoner of the specified dimensions
		@param y: The size of the simulation in the y axis
		@param x: The size of the simulation in the x axis
		*/
		CUDA_FUNCTION ZonerArrayPixels(int * dims, int* maxDims, bool* A, bool* B);

		/** Destructor 1. Default destructor
		*/
		CUDA_FUNCTION ~ZonerArrayPixels();

		/** Update the local store of which zones are 'dead'/inactive
		@param frame1: y*x frame to be compared against frame2
		@param frame2: y*x frame of cell state to be compared against frame1
		*/
		CUDA_FUNCTION bool virtual updateDeadZones(T* frame1, T* frame2) override;

		/** Get whether the target cell is live (may change in the next frame)
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		CUDA_FUNCTION bool virtual isLive(int y, int x) override;

		/** Get the complete matrix of cell activites (whether a cell's state may change in the next frame)
		*/
		CUDA_FUNCTION bool* getCellActivities();

		/** Set the dimensions of the frame that the zoner is handling
		@param y: The size of the frame in the y dimension
		@param x: The size of the frame in the x dimension
		*/
		CUDA_FUNCTION virtual bool setDimensions(int* dims) override;

		/** Reset all stored cell states, matching with the current frame dimensions
		*/
		CUDA_FUNCTION virtual bool refresh() override;
	};
}
#include "ZonerArrayPixels.inl"