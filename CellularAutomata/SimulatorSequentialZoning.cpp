#include "SimulatorSequentialZoning.h"



SimulatorSequentialZoning::SimulatorSequentialZoning(int y, int x, IRules& rules, IDeadZoneHandler& zoner) : SimulatorSequential(y,x,rules), zoner(zoner)
{
}


SimulatorSequentialZoning::~SimulatorSequentialZoning()
{
}

double SimulatorSequentialZoning::stepForward(int steps) {
	timer.reset();
	if (steps < 0) {
		throw std::runtime_error("The simulation cannnot work backwards");
	}
	else {
		for (int step = 0; step < steps; ++step) {
			copyFrame();
			for (int y = 0; y < y_dim; ++y) {
				for (int x = 0; x < x_dim; ++x) {
					if (zoner.isLive(y, x)) {
						// calculate the new cell value based on the old one
						setCell(y, x, rules.getNextState(*(cellStore.end() - 2), y, x));
					}
				}
			}
		}
	}
	zoner.updateDeadZones(*(cellStore.end() - 2), *(cellStore.end() - 1));
	double elapsed = timer.elapsed();
	elapsedTime += elapsed;
	return elapsed;
}