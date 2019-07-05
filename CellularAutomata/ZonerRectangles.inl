template <typename T>
ZonerRectangles<T>::ZonerRectangles(int stepsForDeath, int distanceForDeath, bool deadOrAlive) : deadOrAlive(deadOrAlive), stepsForDeath(stepsForDeath), distanceForDeath(distanceForDeath)
{

}

template <typename T>
ZonerRectangles<T>::~ZonerRectangles()
{
}

template <typename T>
bool ZonerRectangles<T>::updateDeadZones(std::vector<std::vector<T>> frame1, std::vector<std::vector<T>> frame2) {
	return true;
}

template <typename T>
std::vector<std::tuple<int, int, int, int>> ZonerRectangles<T>::getDeadZones() {
	return std::vector<std::tuple<int, int, int, int>>(1,std::make_tuple(0, 0, 0, 0));
};

template <typename T>
bool ZonerRectangles<T>::isLive(int y, int x) {
	for (auto theTuple : Zones) {
		int ymin, ymax, xmin, xmax;
		std::tie(ymin, ymax, xmin, xmax) = theTuple;
		if (y >= ymin && y <= ymax && x >= xmin && x <= xmax) {
			return false;
		}
	}
	return true;
};