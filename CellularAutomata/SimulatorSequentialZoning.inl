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
		if (steps < 0) {
			throw std::runtime_error("The simulation cannnot work backwards");
		}
		else {
			for (int step = 0; step < steps; ++step) {
				this->copyFrame();
				for (int y = 0; y < this->y_dim; ++y) {
					for (int x = 0; x < this->x_dim; ++x) {
						// only calculate a different state if the cell is near any other changes
						if (zoner.isLive(y, x)) {
							// calculate the new cell value based on the old one
							this->setCell(y, x, this->rules.getNextState(*(this->cellStore.end() - 2), y, x));
						}
					}
				}
			}
		}
		zoner.updateDeadZones(*(this->cellStore.end() - 2), *(this->cellStore.end() - 1));
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template <typename T>
	bool SimulatorSequentialZoning<T>::setDimensions(int y, int x)
	{
		this->y_dim = y;
		this->x_dim = x;
		this->rebuildCellStore();
		zoner.setDimensions(y, x);
		return true;
	}
}