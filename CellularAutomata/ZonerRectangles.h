#pragma once
#include "IDeadZoneHandler.h"
class ZonerRectangles :
	public IDeadZoneHandler
{
protected:
	std::vector < std::tuple<int, int, int, int >> Zones;
	// 0/false = dead, 1/true = alive
	bool deadOrAlive;
	int stepsForDeath, distanceForDeath;
public:
	ZonerRectangles(int stepsForDeath, int distanceForDeath, bool deadOrAlive = 0);
	~ZonerRectangles();
	bool virtual updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) override;
	bool virtual isLive(int y, int x) override;
	std::vector<std::tuple<int, int, int, int>> ZonerRectangles::getDeadZones();
};

