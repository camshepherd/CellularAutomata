#pragma once

#include <tuple>
#include <vector>
class IDeadZoneHandler
{
public:
	IDeadZoneHandler();
	~IDeadZoneHandler();
	void virtual updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) = 0;
	std::vector<std::tuple<int, int, int, int>> virtual getDeadZones() = 0;
	bool virtual isLive(int y, int x) = 0;
};

