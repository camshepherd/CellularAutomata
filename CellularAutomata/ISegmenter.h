#pragma once

#include <vector>
#include <tuple>

//
// Interface defining functionality necessary for a class to segment 
// cell stores, ready for assignment to multiple threads
class ISegmenter
{
public:

	virtual std::vector<std::tuple<int, int, int, int>> segment(int y_dim_, int x_dim_, int numSegments) const = 0;
};

