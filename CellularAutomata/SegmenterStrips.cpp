#include "SegmenterStrips.h"



SegmenterStrips::SegmenterStrips(int orientation) : orientation(orientation)
{
}


SegmenterStrips::~SegmenterStrips()
{
}

std::vector<std::tuple<int, int, int, int>> SegmenterStrips::segment(int y_dim, int x_dim, int numSegments) const{

	std::vector<std::tuple<int, int, int, int>> partitions{};

	if (orientation == 0){
		int y_size = y_dim / numSegments;
		for (int k = 0; k < numSegments - 1; ++k){
			partitions.push_back(std::tuple<int, int, int, int>(k * y_size,(k+1) * y_size - 1, 0,x_dim - 1));
		}
		partitions.push_back(std::tuple<int, int, int, int>(numSegments * y_size, y_dim - 1, 0, x_dim - 1));
	}
	else if (orientation == 1) {
		int x_size = x_dim / numSegments;
		for (int k = 0; k < numSegments - 1; ++k) {
			partitions.push_back(std::tuple<int, int, int, int>(0, y_dim - 1, k * x_size, (k + 1) * x_size - 1));
		}
		partitions.push_back(std::tuple<int, int, int, int>(0, y_dim - 1, numSegments * x_size, x_dim - 1));
	}
	
	return partitions;
}