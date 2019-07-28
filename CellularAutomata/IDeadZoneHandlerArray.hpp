#pragma once

#include <tuple>
#include <vector>
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
		IDeadZoneHandlerArray() {};

		/** Destructor 1. Default destructor
		*/
		~IDeadZoneHandlerArray() {};

		/** Update the internal state of regions of activity
		@param frame1: The first frame; to be compared against frame2
		@param frame2: The second frame; to be compared against frame1
		*/
		bool virtual updateDeadZones(T* frame1, T* frame2) = 0;

		/** Whether the targeted cell has the potential to change state in the next frame
		@param y: The cell's y-coordinate
		@param x: The cell's x-coordinate
		*/
		bool virtual isLive(int y, int x) = 0;
	};
}