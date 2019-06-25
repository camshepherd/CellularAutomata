#include "ZonerPixels.h"



ZonerPixels::ZonerPixels(int y, int x) : ydim(y),xdim(x)
{
	cellActivities = std::vector<std::vector<bool>>(y, std::vector<bool>(x, 1));
}


ZonerPixels::~ZonerPixels()
{
}

bool ZonerPixels::updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) {
	return true;
};

bool ZonerPixels::isLive(int y, int x) {
	return true;
};