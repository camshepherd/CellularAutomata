#include "Zoner.h"



Zoner::Zoner(int stepsForDeath, int distanceForDeath, bool deadOrAlive) : deadOrAlive(deadOrAlive), stepsForDeath(stepsForDeath), distanceForDeath(distanceForDeath)
{

}


Zoner::~Zoner()
{
}

void updateDeadZones(std::vector<std::vector<std::vector<int>>> frames) {

}
std::vector<std::tuple<int, int, int, int>> getDeadZones() {
	return std::vector<std::tuple<int, int, int, int>>(1,std::make_tuple(0, 0, 0, 0));
};

bool isLive(int y, int x) {
	return true;
};