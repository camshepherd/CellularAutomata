#include "SegmenterStrips.h"


namespace CellularAutomata {
	SegmenterStrips::SegmenterStrips(bool orientation) : orientation(orientation)
	{
	}


	SegmenterStrips::~SegmenterStrips()
	{
	}

	std::vector<std::tuple<int, int, int, int>> SegmenterStrips::segment(int y_dim, int x_dim, int numSegments) const {

		std::vector<std::tuple<int, int, int, int>> partitions{};
		int partitionSize, excess;
		if (orientation == 0) {
			partitionSize = y_dim / numSegments; // cast to integer just throws away data after the decimal point
		}
		else if (orientation == 1) {
			partitionSize = x_dim / numSegments;
		}
		excess = orientation ? x_dim % numSegments : y_dim % numSegments;

		for (int k = 0; k < numSegments; ++k) {
			partitions.push_back(std::make_tuple(
				orientation ? 0 : k * partitionSize + (k < excess ? k : excess),
				orientation ? y_dim - 1 : ((k + 1) * partitionSize) - 1 + (k < excess ? k + 1 : excess),
				orientation ? k * partitionSize + (k < excess ? k : excess) : 0,
				orientation ? ((k + 1) * partitionSize) - 1 + (k < excess ? k + 1 : excess) : x_dim - 1)
			);
		}

		return partitions;
	}
}