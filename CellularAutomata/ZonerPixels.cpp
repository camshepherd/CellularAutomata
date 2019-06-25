#include "ZonerPixels.h"



ZonerPixels::ZonerPixels(int y, int x) : ydim(y),xdim(x)
{
	cellActivities = std::vector<std::vector<bool>>(y, std::vector<bool>(x, 1));
}


ZonerPixels::~ZonerPixels()
{
}

bool ZonerPixels::updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) {
	// get all cells that are different between the cells
	// mark all differing cell locations, and their neighbours, as being active

	std::vector<std::vector<bool>> rawActivities(ydim, std::vector<bool>(xdim, false));
	
	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			rawActivities[y][x] = frames[0][y][x] != frames[1][y][x];
			cellActivities[y][x] = false;
		}
	}

	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			if (rawActivities[y][x] == true) {
				for (int ypos = y - 1; ypos <= y + 1; ++ypos) {
					for (int xpos = x - 1; xpos <= x + 1; ++xpos) {
						cellActivities[(ypos + ydim) % ydim][(xpos + xdim) % xdim] = true;
					}
				}
			}
		}
	}
	return true;
};

bool ZonerPixels::isLive(int y, int x) {
	return cellActivities[y][x];
};

std::vector<std::vector<bool>> ZonerPixels::getCellActivities() {
	return cellActivities;
}