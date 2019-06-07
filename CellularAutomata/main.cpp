#include "SimulatorConway.h"
#include "RulesConway.h"
int main() {
	RulesConway conway_rules = RulesConway();
	SimulatorConway conway(5, 5, conway_rules);
	conway.setCell(4, 3, 1);
	conway.setCell(4, 2, 1);
	conway.setCell(3, 3, 1);
	conway.stepForward();
	conway.stepForward();
}