#include "stdafx.h"
#include "CppUnitTest.h"

#include "SegmenterStrips.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace SegmenterTesting {
	TEST_CLASS(SegmenterStripsTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		SegmenterStrips thing{ 1 };
		SegmenterStrips thing2{};
		Assert::IsTrue(true);
	}

	TEST_METHOD(CanHandleYSplitting) {
		SegmenterStrips seg{0};
		std::vector<std::tuple<int, int, int, int>> splits;
		
		// even distribution
		splits = seg.segment(4, 4, 2);
		
		Assert::AreEqual(std::get<0>(splits[0]), 0);
		Assert::AreEqual(std::get<1>(splits[0]), 1);
		Assert::AreEqual(std::get<2>(splits[0]), 0);
		Assert::AreEqual(std::get<3>(splits[0]), 3);

		Assert::AreEqual(std::get<0>(splits[1]), 2);
		Assert::AreEqual(std::get<1>(splits[1]), 3);
		Assert::AreEqual(std::get<2>(splits[1]), 0);
		Assert::AreEqual(std::get<3>(splits[1]), 3);

		// uneven distribution
		splits = seg.segment(3, 4, 2);
		Assert::AreEqual(std::get<0>(splits[0]), 0);
		Assert::AreEqual(std::get<1>(splits[0]), 1);
		Assert::AreEqual(std::get<2>(splits[0]), 0);
		Assert::AreEqual(std::get<3>(splits[0]), 3);

		Assert::AreEqual(std::get<0>(splits[1]), 2);
		Assert::AreEqual(std::get<1>(splits[1]), 2);
		Assert::AreEqual(std::get<2>(splits[1]), 0);
		Assert::AreEqual(std::get<3>(splits[1]), 3);

		// handle many segments
		splits = seg.segment(10, 6, 4);
		Assert::AreEqual(std::get<0>(splits[0]), 0);
		Assert::AreEqual(std::get<1>(splits[0]), 2);
		Assert::AreEqual(std::get<2>(splits[0]), 0);
		Assert::AreEqual(std::get<3>(splits[0]), 5);
	}

	TEST_METHOD(CanHandleXSplitting) {
		SegmenterStrips seg{ 1 };
		std::vector<std::tuple<int, int, int, int>> splits;

		// even distribution
		splits = seg.segment(4, 4, 2);

		Assert::AreEqual(std::get<0>(splits[0]), 0);
		Assert::AreEqual(std::get<1>(splits[0]), 3);
		Assert::AreEqual(std::get<2>(splits[0]), 0);
		Assert::AreEqual(std::get<3>(splits[0]), 1);

		Assert::AreEqual(std::get<0>(splits[1]), 0);
		Assert::AreEqual(std::get<1>(splits[1]), 3);
		Assert::AreEqual(std::get<2>(splits[1]), 2);
		Assert::AreEqual(std::get<3>(splits[1]), 3);

		// uneven distribution
		splits = seg.segment(4, 3, 2);
		Assert::AreEqual(std::get<0>(splits[0]), 0);
		Assert::AreEqual(std::get<1>(splits[0]), 3);
		Assert::AreEqual(std::get<2>(splits[0]), 0);
		Assert::AreEqual(std::get<3>(splits[0]), 1);

		Assert::AreEqual(std::get<0>(splits[1]), 0);
		Assert::AreEqual(std::get<1>(splits[1]), 3);
		Assert::AreEqual(std::get<2>(splits[1]), 2);
		Assert::AreEqual(std::get<3>(splits[1]), 2);

		// handle many segments
		splits = seg.segment(6, 10, 4);
		Assert::AreEqual(std::get<0>(splits[0]), 0);
		Assert::AreEqual(std::get<1>(splits[0]), 5);
		Assert::AreEqual(std::get<2>(splits[0]), 0);
		Assert::AreEqual(std::get<3>(splits[0]), 2);
	}
		
	};
}