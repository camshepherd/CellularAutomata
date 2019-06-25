#include "ZonerPixels.h"



ZonerPixels::ZonerPixels(int y, int x) : ydim(y),xdim(x)
{
	cellActivities = std::vector<std::vector<bool>>(y, std::vector<bool>(x, 1));
}


ZonerPixels::~ZonerPixels()
{
}

bool ZonerPixels::updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) {
	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			cellActivities[y][x] = frames[0][y][x] != frames[1][y][x];
		}
	}

	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			if (cellActivities[y][x]) {
				for (int ypos = y - 1; ypos <= y + 1; ypos) {
					for (int xpos = x - 1; xpos <= x + 1; ++xpos) {
						cellActivities[y][x] = true;
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
