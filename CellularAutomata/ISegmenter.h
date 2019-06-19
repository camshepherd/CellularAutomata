#pragma once

#include <vector>
class ISegmenter
{
public:
	ISegmenter();
	~ISegmenter();

	virtual std::vector<std::tuple<int, int, int, int>>& segment(int y_dim_, int x_dim_, int numSegments) const = 0;
};

