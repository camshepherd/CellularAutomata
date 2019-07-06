#include "stdafx.h"
#include "CppUnitTest.h"

#include <ZonerPixels.hpp>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace ZonerTesting {
	TEST_CLASS(ZonerPixelsTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		ZonerPixels<int> deadZone{ 3,3 };
		deadZone.isLive(1,1);
		Assert::IsTrue(true);
	}

	TEST_METHOD(UpdateMinDimensions) {
		
		// no changes
		ZonerPixels<int> zoner{ 3,3 };
		std::vector<std::vector<int>> frame1(3, std::vector<int>(3, 0));
		std::vector<std::vector<int>> frame2(frame1.begin(), frame1.end());

		zoner.updateDeadZones(frame1, frame2);
		std::vector<std::vector<bool>> expected(3,std::vector<bool>(3,false));
		auto actual = zoner.getCellActivities();

		for (int y = 0; y < 3; ++y) {
			for (int x = 0; x < 3; ++x) {
				Assert::IsTrue(expected[y][x] == actual[y][x]);
			}
		}

		// single change at [0, 0]
		frame1[0][0] = 1;

		zoner.updateDeadZones(frame1, frame2);

		expected = std::vector<std::vector<bool>>(3, std::vector<bool>(3, true));
		actual = zoner.getCellActivities();

		for (int y = 0; y < 3; ++y) {
			for (int x = 0; x < 3; ++x) {
				Assert::IsTrue(expected[y][x] == actual[y][x]);
			}
		}
	}


	TEST_METHOD(UpdateMoreDimensions) {
		// single change

		ZonerPixels<int> zoner{ 6,4 };
		std::vector<std::vector<int>> frame1(6, std::vector<int>(4, 0));
		std::vector<std::vector<int>> frame2(frame1.begin(), frame1.end());

		frame1[1][1] = 1;

		zoner.updateDeadZones(frame1, frame2);
		std::vector<std::vector<bool>> expected(6, std::vector<bool>(4, false));

		expected[0][1] = true;
		expected[1][1] = true;
		expected[2][1] = true;
		expected[1][0] = true;
		expected[1][2] = true;
		expected[0][0] = true;
		expected[2][2] = true;
		expected[0][2] = true;
		expected[2][0] = true;

		std::vector<std::vector<bool>> actual(6, std::vector<bool>(4, false));
		actual = zoner.getCellActivities();

		for (int y = 0; y < 6; ++y) {
			for (int x = 0; x < 4; ++x) {
				Assert::IsTrue(expected[y][x] == actual[y][x]);
			}
		}
	}
	};
}
