#include "SegmenterLinear.h"



SegmenterLinear::SegmenterLinear(int orientation, int dominantRatio) : orientation(orientation), dominantRatio(dominantRatio)
{
}


SegmenterLinear::~SegmenterLinear()
{
}

std::vector<std::tuple<int, int, int, int>>& SegmenterLinear::segment(int y_dim_, int x_dim_, int numSegments) const{
	int y_dim = y_dim_ ? orientation : x_dim_;
	int x_dim = x_dim ? orientation : y_dim_;

	int y_split = y_dim / dominantRatio;
}