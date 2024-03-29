namespace CellularAutomata {
	template<typename T>
	bool ISimulator<T>::writeData(std::string fileName) {
		std::ofstream file;
		try {
			file.open(fileName);
			file << elapsedTime << "\n";
			file << getYDim() << "," << getXDim() << "," << getNumFrames() << "\n";
			for (int t = 0; t < getNumFrames(); ++t) {
				for (int y = 0; y < getYDim(); ++y) {
					for (int x = 0; x < getXDim(); x++) {
						file << getCell(y, x, t) << ",";
					}
					file << "\n";
				}
			}
			file.close();
		}
		catch (std::exception e) {
			file.close();
			return false;
		}
		return true;
	}

	template <typename T>
	double ISimulator<T>::stepForwardTime(double seconds) {
		double elapsed = 0;
		while (elapsed <= seconds) {
			elapsed += stepForward();
		}
		elapsedTime += elapsed;
		return elapsed;
	}
}