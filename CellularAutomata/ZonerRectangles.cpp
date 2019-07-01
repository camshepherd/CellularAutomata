#include "ZonerRectangles.h"



ZonerRectangles::ZonerRectangles(int stepsForDeath, int distanceForDeath, bool deadOrAlive) : deadOrAlive(deadOrAlive), stepsForDeath(stepsForDeath), distanceForDeath(distanceForDeath)
{

}


ZonerRectangles::~ZonerRectangles()
{
}

bool ZonerRectangles::updateDeadZones(std::vector<std::vector<int>> frame1, std::vector<std::vector<int>> frame2) {
	return true;
}
std::vector<std::tuple<int, int, int, int>> ZonerRectangles::getDeadZones() {
	return std::vector<std::tuple<int, int, int, int>>(1,std::make_tuple(0, 0, 0, 0));
};

bool ZonerRectangles::isLive(int y, int x) {
	for (auto theTuple : Zones) {
		int ymin, ymax, xmin, xmax;
		std::tie(ymin, ymax, xmin, xmax) = theTuple;
		if (y >= ymin && y <= ymax && x >= xmin && x <= xmax) {
			return false;
		}
	}
	return true;
};