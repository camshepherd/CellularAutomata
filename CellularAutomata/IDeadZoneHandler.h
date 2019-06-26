#pragma once

#include <tuple>
#include <vector>
class IDeadZoneHandler
{
public:
	IDeadZoneHandler() {};
	~IDeadZoneHandler() {};
	bool virtual updateDeadZones(std::vector<std::vector<int>> frame1, std::vector<std::vector<int>> frame2) = 0;
	bool virtual isLive(int y, int x) = 0;
};

