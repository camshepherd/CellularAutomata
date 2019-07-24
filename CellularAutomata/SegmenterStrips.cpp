#include "SegmenterStrips.hpp"


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
			partitionSize = y_dim / numSegments; // implicit cast to integer just throws away data after the decimal point
		}
		else if (orientation == 1) {
			partitionSize = x_dim / numSegments;
		}
		else
		{
			throw std::runtime_error("Invalid orientation encountered");
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

	int* SegmenterStrips::segmentToArray(int y_dim, int x_dim, int numSegments) const {
		int* partitions = static_cast<int*>(malloc(sizeof(int) * (numSegments * 4)));
		bool ended = false;
		// store in the array the dimensions of it
		int partitionSize, excess;
		if (orientation == 0) {
			partitionSize = y_dim / numSegments; // implicit cast to integer just throws away data after the decimal point
		}
		else if (orientation == 1) {
			partitionSize = x_dim / numSegments;
		}
		else
		{
			throw std::runtime_error("Invalid orientation encountered");
		}
		excess = orientation ? x_dim % numSegments : y_dim % numSegments;

		for (int k = 0; k < numSegments; ++k) {
			partitions[k*4] = orientation ? 0 : k * partitionSize + (k < excess ? k : excess);
			partitions[k*4 + 1] = orientation ? y_dim - 1 : ((k + 1) * partitionSize) - 1 + (k < excess ? k + 1 : excess);
			partitions[k*4 + 2] = orientation ? k * partitionSize + (k < excess ? k : excess) : 0;
			partitions[k*4 + 3] = orientation ? ((k + 1) * partitionSize) - 1 + (k < excess ? k + 1 : excess) : x_dim - 1;

			if(ended || (partitions[k*4] == partitions[(k-1) * 4] && partitions[k*4 + 2] == partitions[(k-1)*4 + 2]))
			{
				ended = true;
				// if it is not feasible to use all segments, then end
				partitions[k * 4] = -1;
				partitions[k * 4 + 1] = -1;
				partitions[k * 4 + 2] = -1;
				partitions[k * 4 + 3] = -1;
			}
		}

		printf("Segmenter: \n");
		for(int m = 0; m < numSegments * 4; m += 4)
		{
			for(int n = 0; n < 4; ++n)
			{
				printf("%d,", partitions[m + n]);
			}
		}

		return partitions;
	}
}