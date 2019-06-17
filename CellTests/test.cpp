#include "pch.h"
#include "RulesConway.h"
#include "SimulatorSequential.h"

TEST(SyntaxTest, DefaultTest) {
  EXPECT_EQ(1, 1);
  EXPECT_TRUE(true);
}

TEST(Rules,Conway) {
	// Using default rules: birth number = 3, stay alive with 2 or 3 neighbours
	RulesConway conway = RulesConway();

	std::vector<std::vector<int>> frame(3,std::vector<int>());
	for (auto thing : frame) {
		thing = std::vector<int>(3,0);
	}
	EXPECT_EQ(frame[0][0], 0);
	EXPECT_EQ(frame[2][2], 0);
	

	// Test total absences and wrap-around
	EXPECT_EQ(conway.getNextState(frame, 0, 0), 0);
	EXPECT_EQ(conway.getNextState(frame, 1, 1), 0);
	EXPECT_EQ(conway.getNextState(frame, 2, 2), 0);
	EXPECT_EQ(conway.getNextState(frame, 2, 0), 0);
	EXPECT_EQ(conway.getNextState(frame, 0, 2), 0);
	EXPECT_EQ(conway.getNextState(frame, 1, 0), 0);

	//Test with single alive
	frame[0][0] = 1; //alive
	EXPECT_EQ(conway.getNextState(frame, 0, 0), 0);
	EXPECT_EQ(conway.getNextState(frame, 0, 1), 0);
	EXPECT_EQ(conway.getNextState(frame, 2, 2), 0);
	EXPECT_EQ(conway.getNextState(frame, 2, 0), 0);

	//Test with two alive
	frame[0][1] = 1;
	EXPECT_EQ(conway.getNextState(frame, 0, 0), 1);
	EXPECT_EQ(conway.getNextState(frame, 0, 1), 1);
	EXPECT_EQ(conway.getNextState(frame, 1, 0), 0);
	EXPECT_EQ(conway.getNextState(frame, 2, 2), 0);
	EXPECT_EQ(conway.getNextState(frame, 0, 2), 0);
	EXPECT_EQ(conway.getNextState(frame, 2, 0), 0);

	//Test three alive
	frame[2][2] = 1;
	//alive
	EXPECT_EQ(conway.getNextState(frame, 0, 1), 1);
	EXPECT_EQ(conway.getNextState(frame, 0, 0), 1);
	EXPECT_EQ(conway.getNextState(frame, 2, 2), 1);
	//dead
	EXPECT_EQ(conway.getNextState(frame, 2, 0), 1);
	EXPECT_EQ(conway.getNextState(frame, 0, 2), 1);
	EXPECT_EQ(conway.getNextState(frame, 1, 2), 1);

	//Test Four alive
	frame[2][1] = 1;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_EQ(conway.getNextState(frame, y, x), 0);
		}
	}

	//Test five alive
	frame[2][0] = 1;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_EQ(conway.getNextState(frame, y, x), 0);
		}
	}

	// Test six alive
	frame[0][2] = 1;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_EQ(conway.getNextState(frame, y, x), 0);
		}
	}

	//test seven alive
	frame[1][0] = 1;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_EQ(conway.getNextState(frame, y, x), 0);
		}
	}

	// Test eight alive
	frame[1][1] = 1;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_EQ(conway.getNextState(frame, y, x), 0);
		}
	}

	//Test nine alive
	frame[1][2] = 1;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_EQ(conway.getNextState(frame, y, x), 0);
		}
	}
}

TEST(Simulator, SequentialConway) {
	RulesConway conway = RulesConway();
	SimulatorSequential sim(5, 5, conway);
	EXPECT_EQ(sim.getNumFrames(), 1);
	//Test initial state
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			EXPECT_EQ(sim.getCell(y, x), 0);
		}
	}

	// Test blank frame
	sim.blankFrame();
	EXPECT_EQ(sim.getNumFrames(), 2);
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			EXPECT_EQ(sim.getCell(y, x, 1), 0);
		}
	}

	sim.blankFrame();
	EXPECT_EQ(sim.getNumFrames(), 3);
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			EXPECT_EQ(sim.getCell(y, x, 2), 0);
		}
	}


	// test setting/getting cell with default time
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			sim.setCell(y, x, 1);
		}
	}
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			EXPECT_EQ(sim.getCell(y, x), 1);
		}
	}

	// test setting/getting cell with explicit time
	sim.blankFrame();
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			sim.setCell(y, x, 3,1);
		}
	}
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			EXPECT_EQ(sim.getCell(y, x, 3), 1);
		}
	}

	EXPECT_EQ(sim.getNumFrames(), 4);
}