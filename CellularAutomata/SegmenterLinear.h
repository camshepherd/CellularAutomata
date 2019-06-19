#pragma once
#include "ISegmenter.h"
class SegmenterLinear :
	public ISegmenter
{
private:
	int orientation, dominantRatio;

public:
	SegmenterLinear(int orientation = 0, int dominantRatio = 1);
	~SegmenterLinear();

	std::vector<std::tuple<int, int, int, int>>& segment(int y_dim_, int x_dim_, int numSegments) const override;
};

