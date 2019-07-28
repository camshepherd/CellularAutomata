#include "stdafx.h"
#include "CppUnitTest.h"

#include <ZonerArrayPixels.hpp>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace CellularAutomata;
namespace ZonerTesting {
	TEST_CLASS(ZonerArrayPixelsTesting) {
public:
	TEST_METHOD(CanInstantiate) {
		ZonerArrayPixels<int> deadZone{ 3,3 };
		deadZone.isLive(1, 1);
		Assert::IsTrue(true);
	}

	TEST_METHOD(UpdateMinDimensions) {

		// no changes
		ZonerArrayPixels<int> zoner{ 3,3 };

		int *frame1 = static_cast<int*>(malloc(sizeof(int) * 3 * 3)), 
			*frame2 = static_cast<int*>(malloc(sizeof(int) * 3 * 3));
		bool *expected = static_cast<bool*>(malloc(sizeof(bool) * 3 * 3)),
			*actual = static_cast<bool*>(malloc(sizeof(bool) * 3 * 3));
		for(int t = 0; t < 9; ++t)
		{
			frame1[t] = 0;
			frame2[t] = 0;
			expected[t] = false;
		}

		zoner.updateDeadZones(frame1, frame2);

		
		actual = zoner.getCellActivities();

		for (int y = 0; y < 3; ++y) {
			for (int x = 0; x < 3; ++x) {
				Assert::IsTrue(expected[y*3 + x] == actual[y*3 + x]);
			}
		}

		// single change at [0, 0]
		frame1[0] = 1;

		zoner.updateDeadZones(frame1, frame2);

		for(int r = 0; r < 9; ++r)
		{
			expected[r] = true;
		}
		actual = zoner.getCellActivities();

		for (int y = 0; y < 3; ++y) {
			for (int x = 0; x < 3; ++x) {
				Assert::IsTrue(expected[y*3 + x] == actual[y * 3 + x]);
			}
		}
	}


	TEST_METHOD(UpdateMoreDimensions) {
		// single change

		ZonerArrayPixels<int> zoner{ 6,4 };
		int *frame1 = static_cast<int*>(malloc(sizeof(int) * 6 * 4)),
			*frame2 = static_cast<int*>(malloc(sizeof(int) * 6 * 4));
		bool *expected = static_cast<bool*>(malloc(sizeof(bool) * 6 * 4)),
			*actual;

		for (int t = 0; t < 6*4; ++t)
		{
			frame1[t] = 0;
			frame2[t] = 0;
			expected[t] = false;
		}

		frame1[5] = 1;

		zoner.updateDeadZones(frame1, frame2);

		expected[1] = true;
		expected[1*4+1] = true;
		expected[2*4+1] = true;
		expected[1*4+0] = true;
		expected[1*4+2] = true;
		expected[0] = true;
		expected[2*4+2] = true;
		expected[2] = true;
		expected[2*4+0] = true;

		actual = zoner.getCellActivities();

		Assert::IsTrue(actual[1]);
		Assert::IsTrue(actual[5]);
		Assert::IsTrue(actual[9]);
		Assert::IsTrue(actual[4]);
		Assert::IsTrue(actual[6]);
		Assert::IsTrue(actual[0]);
		Assert::IsTrue(actual[10]);
		Assert::IsTrue(actual[2]);
		Assert::IsTrue(actual[8]);


		for (int y = 0; y < 6; ++y) {
			for (int x = 0; x < 4; ++x) {
				if(expected[y*4+x] != actual[y*4+x])
				{
					printf("FAILED AT %d, %d", y, x);
				}
				Assert::IsTrue(expected[y*4+x] == actual[y*4+x]);
			}
		}
	}
	};
}
