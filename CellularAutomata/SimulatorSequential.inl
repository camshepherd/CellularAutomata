template <typename T>
SimulatorSequential<T>::SimulatorSequential(const int y, const int x, const IRules<T>& _rules) : SimulatorVector<T>(y,x,_rules)
{
}

template <typename T>
SimulatorSequential<T>::~SimulatorSequential()
{
}

template <typename T>
double SimulatorSequential<T>::stepForward(int steps) {
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

template <typename T>
T SimulatorSequential<T>::getMaxValidState() {
	return rules.getMaxValidState();
}