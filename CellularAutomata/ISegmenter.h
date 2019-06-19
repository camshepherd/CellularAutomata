#pragma once

#include <vector>
#include <tuple>
class ISegmenter
{
public:

	virtual std::vector<std::tuple<int, int, int, int>> segment(int y_dim_, int x_dim_, int numSegments) const = 0;
};

