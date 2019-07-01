#include "SimulatorSequential.h"

SimulatorSequential::SimulatorSequential(const int y, const int x, const IRules& _rules) : SimulatorVector(y,x,_rules)
{
}


SimulatorSequential::~SimulatorSequential()
{
}

double SimulatorSequential::stepForward(int steps) {
	timer.reset();
	if (steps < 0) {
		throw std::runtime_error("The simulation cannnot work backwards");
	}
	else {
		for (int step = 0; step < steps; ++step) {
			blankFrame();
			for (int y = 0; y < y_dim; ++y) {
				for (int x = 0; x < x_dim; ++x) {
					// calculate the new cell value based on the old one
					setCell(y, x, rules.getNextState(*(cellStore.end() - 2), y, x));
				}
			}
		}
	}
	double elapsed = timer.elapsed();
	elapsedTime += elapsed;
	return elapsed;
}
