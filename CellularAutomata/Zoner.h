#pragma once
#include "IDeadZoneHandler.h"
class Zoner :
	public IDeadZoneHandler
{
protected:
	std::vector < std::tuple<int, int, int, int >> Zones;
	// 0/false = dead, 1/true = alive
	bool deadOrAlive;
	int stepsForDeath, distanceForDeath;
	bool isInDeadZone(int y, int x);
public:
	Zoner(int stepsForDeath, int distanceForDeath, bool deadOrAlive = 0);
	~Zoner();
	void virtual updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) override;
	std::vector<std::tuple<int, int, int, int>> virtual getDeadZones() override;
	bool virtual isLive(int y, int x) override;
};

