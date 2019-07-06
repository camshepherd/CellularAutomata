namespace CellularAutomata {
	template <typename T>
	SimulatorSequential<T>::SimulatorSequential(const int y, const int x, const IRules<T>& _rules) : SimulatorVector<T>(y, x, _rules)
	{
	}

	template <typename T>
	SimulatorSequential<T>::~SimulatorSequential()
	{
	}

	template <typename T>
	double SimulatorSequential<T>::stepForward(int steps) {
		this->timer.reset();
		if (steps < 0) {
			throw std::runtime_error("The simulation cannnot work backwards");
		}
		else {
			for (int step = 0; step < steps; ++step) {
				this->blankFrame();
				for (int y = 0; y < this->y_dim; ++y) {
					for (int x = 0; x < this->x_dim; ++x) {
						// calculate the new cell value based on the old one
						this->setCell(y, x, this->rules.getNextState(*(this->cellStore.end() - 2), y, x));
					}
				}
			}
		}
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template <typename T>
	T SimulatorSequential<T>::getMaxValidState() {
		return this->rules.getMaxValidState();
	}
}