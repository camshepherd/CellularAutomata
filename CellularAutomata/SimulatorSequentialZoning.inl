namespace CellularAutomata {
	template <typename T>
	SimulatorSequentialZoning<T>::SimulatorSequentialZoning(int y, int x, IRules<T>& rules, IDeadZoneHandler<T>& zoner) : SimulatorSequential<T>(y, x, rules), zoner(zoner)
	{
	}

	template <typename T>
	SimulatorSequentialZoning<T>::~SimulatorSequentialZoning()
	{
	}

	template <typename T>
	double SimulatorSequentialZoning<T>::stepForward(int steps) {
		timer.reset();
		if (steps < 0) {
			throw std::runtime_error("The simulation cannnot work backwards");
		}
		else {
			for (int step = 0; step < steps; ++step) {
				copyFrame();
				for (int y = 0; y < y_dim; ++y) {
					for (int x = 0; x < x_dim; ++x) {
						// only calculate a different state if the cell is near any other changes
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
}