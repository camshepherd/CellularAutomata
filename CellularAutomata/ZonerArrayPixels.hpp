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
		bool* cellActivities, *rawActivities;
		
	public:
		int ydim, xdim;
		/** Constructor 1. Create a zoner of the specified dimensions
		@param y: The size of the simulation in the y axis
		@param x: The size of the simulation in the x axis
		*/
		CUDA_FUNCTION ZonerArrayPixels(int y, int x, bool* A, bool* B);

		/** Destructor 1. Default destructor
		*/
		CUDA_FUNCTION ~ZonerArrayPixels();

		/** Update the local store of which zones are 'dead'/inactive
		@param frame1: y*x frame to be compared against frame2
		@param frame2: y*x frame of cell state to be compared against frame1
		*/
		CUDA_FUNCTION bool virtual updateDeadZones(T* frame1, T* frame2,int* dimensions) override;

		/** Get whether the target cell is live (may change in the next frame)
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		*/
		CUDA_FUNCTION bool virtual isLive(int y, int x) override;

		/** Get the complete matrix of cell activites (whether a cell's state may change in the next frame)
		*/
		CUDA_FUNCTION bool* getCellActivities();

		CUDA_FUNCTION virtual bool setDimensions(int y, int x) override;
	};
}
#include "ZonerArrayPixels.inl"