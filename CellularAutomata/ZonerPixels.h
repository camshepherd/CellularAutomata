#pragma once
#include "IDeadZoneHandler.h"
class ZonerPixels :
	public IDeadZoneHandler
{
protected:
	std::vector<std::vector<bool>> cellActivities;
	int ydim, xdim;
public:
	ZonerPixels(int y, int x);
	~ZonerPixels();
	bool virtual updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) override;
	bool virtual isLive(int y, int x) override;
	std::vector<std::vector<bool>> getCellActivities();
};
