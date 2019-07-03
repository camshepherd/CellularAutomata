#pragma once
#include "IDeadZoneHandler.h"


/** Class to keep track of how rectangular portions of the map may change from frame to frame in the simulation.
*/
class ZonerRectangles :
	public IDeadZoneHandler
{
protected:
	std::vector < std::tuple<int, int, int, int >> Zones;
	// 0/false = dead, 1/true = alive
	bool deadOrAlive;
	int stepsForDeath, distanceForDeath;
public:
	/** Constructor 1.
	*/
	ZonerRectangles(int stepsForDeath, int distanceForDeath, bool deadOrAlive = 0);
	
	/** Destructor 1. Default destructor
	*/
	~ZonerRectangles();

	/** Update the local store of which zones are 'dead'/inactive
	@param frame1: y*x frame to be compared against frame2
	@param frame2: y*x frame of cell state to be compared against frame1
	*/
	bool virtual updateDeadZones(std::vector<std::vector<int>> frame1, std::vector<std::vector<int>> frame2) override;

	/** Get whether the target cell is live (may change in the next frame)
	@param y: The y-coordinate of the target cell
	@param x: The x-coordinate of the target cell
	*/
	bool virtual isLive(int y, int x) override;

	/** Get the complete matrix of cell activites (whether a cell's state may change in the next frame)
	*/
	std::vector<std::tuple<int, int, int, int>> ZonerRectangles::getDeadZones();
};

