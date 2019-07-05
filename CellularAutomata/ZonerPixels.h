#pragma once
#include "IDeadZoneHandler.h"


/** Class to keep track of which pixels may change from frame to frame, and hence require simulation
*/
template <typename T>
class ZonerPixels :
	public IDeadZoneHandler<T>
{
protected:
	std::vector<std::vector<bool>> cellActivities;
	int ydim, xdim;
public:
	/** Constructor 1. Create a zoner of the specified dimensions
	@param y: The size of the simulation in the y axis
	@param x: The size of the simulation in the x axis
	*/
	ZonerPixels(int y, int x);

	/** Destructor 1. Default destructor
	*/
	~ZonerPixels();

	/** Update the local store of which zones are 'dead'/inactive
	@param frame1: y*x frame to be compared against frame2
	@param frame2: y*x frame of cell state to be compared against frame1
	*/
	bool virtual updateDeadZones(std::vector<std::vector<T>> frame1, std::vector<std::vector<T>> frame2) override;

	/** Get whether the target cell is live (may change in the next frame)
	@param y: The y-coordinate of the target cell
	@param x: The x-coordinate of the target cell
	*/
	bool virtual isLive(int y, int x) override;

	/** Get the complete matrix of cell activites (whether a cell's state may change in the next frame)
	*/
	std::vector<std::vector<bool>> getCellActivities();
};

#include "ZonerPixels.inl"