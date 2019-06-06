#include "SimulatorConway.h"

int main() {
	SimulatorConway conway = SimulatorConway(5,5);
	conway.setCell(4, 3, 1);
	conway.setCell(4, 2, 1);
	conway.setCell(3, 3, 1);
	conway.stepForward();
	conway.stepForward();
}