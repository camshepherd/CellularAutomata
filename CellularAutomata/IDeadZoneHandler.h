#pragma once

#include <tuple>
#include <vector>
class IDeadZoneHandler
{
public:
	IDeadZoneHandler() {};
	~IDeadZoneHandler() {};
	bool virtual updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) = 0;
	bool virtual isLive(int y, int x) = 0;
};

