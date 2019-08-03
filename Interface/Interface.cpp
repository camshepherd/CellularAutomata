// Interface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <time.h>

using namespace std;
using namespace CellularAutomata;

bool running = true;
int simType = 0;

ISimulator<bool>* simBool;
ISimulator<int>* simInt;
ISimulator<long>* simLong;
ISimulator<long long>* simLongLong;

IRules<bool>* rulesBool;
IRules<int>* rulesInt;
IRules<long>* rulesLong;
IRules<long long>* rulesLongLong;

ISegmenter* segmenter;

IDeadZoneHandler<int>* zoner;


void printInsult() {
	int num = std::rand() / ((RAND_MAX + 1u) / 6);
	switch (num) {
	case 0:
		cout << "Your mother was a hamster and your father smelt of elderberries!" << endl;
		break;
	case 1:
		cout << "I fart in your general direction!" << endl;
		break;
	case 2:
		cout << "You great supine protoplasmic invertebrate jelly!" << endl;
		break;
	case 3:
		cout << "There was a young fellow from Ankara\n\nWho was a terrific wankerer\n\nTill he sowed his wild oats\n\nWith the help of a goat\n\nBut he didn't even stop to thankera.\n\n";
		break;
	case 4:
		cout << "I am drunk today madam, and tomorrow I shall be sober but you will still be ugly" << endl;
		break;
	case 5:
		cout << "If I have seen further it is by standing on the shoulders of giants" << endl;
		break;
	}
}


void printHelp(int angriness) {
	cout << "\n\n|______________________" << endl;
	cout << std::uppercase << "Help Page/Manual: " << endl;
	cout << "Note that all functionality is very primitive and will assume that you know what you're doing" << endl;
	cout << "print [frameNum] [numToPrint]" << endl;
	cout << "build simulatorType ruleSet [datatype=int] [ydim xdim nSegments nBlocks nThreads ymax xmax]: build simulator with given parameters, not all are needed for all implementations" << endl;
	cout << "dimensions ydim xdim: change dimensions of simulator" << endl;
	cout << "step [steps=1]: step forward through the simulation by the given number of frames" << endl;
	cout << "clear: empties the current simulator" << endl;
	cout << "set y x val: set position [y,x] to value val" << endl;
	cout << "exit: exit the application in a way that doesn't involve crashing" << endl;
	cout << "|__________________________ END\n\n";
}

template <typename T>
void printFrames(ISimulator<T> *sim, int frameStart, int frameCount) {
	for (int t = 0; t < frameCount; ++t) {
		if (frameStart != -1) {
			sim->printFrame(frameStart + t);
		}
		else {
			sim->printFrame();
		}
	}
}


void handleInput(string line) {
	if (line.length() == 0) {
		return;
	}
	vector<string> words{};
	stringstream stream(line);
	for (string word; stream >> word;) {
		words.push_back(word);
	}
	if (words[0] == "help") {
		if (words.size() > 1 && words[1] == "1") {
			printHelp(1);
		}
		else { 
			printHelp(0); 
		}
	}
	else if (words[0] == "insult") {
		printInsult();
	}
	else if (words[0] == "build") {
		int ydim, xdim;
		// build simulatorType ruleSet [datatype=int] [ydim xdim nSegments nBlocks nThreads ymax xmax]
		if (words[1] == "seq") {
			if (words.size() > 3 && words[3] == "int") {
				if (words[2] == "conway" || words[2] == "gol") {
					rulesInt = new RulesConway<int>{};
				}
				else {
					rulesInt = new RulesBML<int>{};
				}
				if (words.size() > 4) {
					ydim = stoi(words[4]);
					xdim = stoi(words[5]);
				}
				simType = 1;
				simInt = new SimulatorSequential<int>{ ydim, xdim, *rulesInt };
				cout << "Created\n";
			}
		}
		else if (words[1] == "seqzon") {
			if (words.size() > 3 && words[3] == "int") {
				if (words[2] == "conway" || words[2] == "gol") {
					rulesInt = new RulesConway<int>{};
				}
				else {
					rulesInt = new RulesBML<int>{};
				}
				if (words.size() > 5) {
					ydim = stoi(words[4]);
					xdim = stoi(words[5]);
				}
				zoner = new ZonerPixels<int>{ ydim,xdim };
				simType = 1;
				simInt = new SimulatorSequentialZoning<int>{ ydim, xdim, *rulesInt,*zoner };
				cout << "Created\n";
			}
		}
		else if (words[1] == "cpuhor") {
			if (words.size() > 3 && words[3] == "int") {
				if (words[2] == "conway" || words[2] == "gol") {
					rulesInt = new RulesConway<int>{};
				}
				else {
					rulesInt = new RulesBML<int>{};
				}
				if (words.size() > 4) {
					ydim = stoi(words[4]);
					xdim = stoi(words[5]);
				}
				segmenter = new SegmenterStrips{ 0 };

				simType = 1;
				simInt = new SimulatorCPU<int>{ ydim, xdim, *rulesInt,*segmenter };
				cout << "Created\n";
			}
		}
		else if (words[1] == "cpuver") {
			if (words.size() > 3 && words[3] == "int") {
				if (words[2] == "conway" || words[2] == "gol") {
					rulesInt = new RulesConway<int>{};
				}
				else {
					rulesInt = new RulesBML<int>{};
				}
				if (words.size() > 4) {
					ydim = stoi(words[4]);
					xdim = stoi(words[5]);
				}
				segmenter = new SegmenterStrips{ 1 };

				simType = 1;
				simInt = new SimulatorCPU<int>{ ydim, xdim, *rulesInt,*segmenter };
				cout << "Created\n";
			}
		}
		else if (words[1] == "cpuhorzon") {
			if (words.size() > 3 && words[3] == "int") {
				if (words[2] == "conway" || words[2] == "gol") {
					rulesInt = new RulesConway<int>{};
				}
				else {
					rulesInt = new RulesBML<int>{};
				}
				if (words.size() > 4) {
					ydim = stoi(words[4]);
					xdim = stoi(words[5]);
				}
				segmenter = new SegmenterStrips{ 0 };

				zoner = new ZonerPixels<int>{ ydim,xdim };
				simType = 1;
				simInt = new SimulatorCPUZoning<int>{ ydim, xdim, *rulesInt,*segmenter,*zoner };
				cout << "Created\n";
			}
		}
	}
	else if (words[0] == "print") {
		// print [frameNum] [numToPrint]
		int frameStart = -1, frameCount = 1;
		if (words.size() > 1){
			frameStart = stoi(words[1]);
		}
		if (words.size() > 2) {
			frameCount = stoi(words[2]);
		}
		if (simType == 0) {
			printFrames<bool>(simBool, frameStart, frameCount);
		}
		else if (simType == 1) {
			printFrames<int>(simInt, frameStart, frameCount);
		}
		else if (simType == 2) {
			printFrames<long>(simLong, frameStart, frameCount);
		}
		else if (simType == 3) {
			printFrames<long long>(simLongLong, frameStart, frameCount);
		}
	}
	else if (words[0] == "step") {
		//step[steps = 1]
		int steps = 1;
		if (words.size() > 1) {
			steps = stoi(words[1]);
		}
		if (simType == 0) {
			simBool->stepForward(steps);
		}
		else if (simType == 1) {
			simInt->stepForward(steps);
		}
		else if (simType == 2) {
			simLong->stepForward(steps);
		}
		else if (simType == 3) {
			simLongLong->stepForward(steps);
		}
	}
	else if (words[0] == "clear") {
		if (simType == 0) {
			simBool->clear();
		}
		else if (simType == 1) {
			simInt->clear();
		}
		else if (simType == 2) {
			simLong->clear();
		}
		else if (simType == 3) {
			simLongLong->clear();
		}
	}
	else if (words[0] == "dims") {
		// dimensions ydim xdim
		int ydim = stoi(words[1]);
		int xdim = stoi(words[2]);
		if (simType == 0) {
			simBool->setDimensions(ydim, xdim);
		}
		else if (simType == 1) {
			simInt->setDimensions(ydim, xdim);
		}
		else if (simType == 2) {
			simLong->setDimensions(ydim, xdim);
		}
		else if (simType == 3) {
			simLongLong->setDimensions(ydim, xdim);
		}
	}
	else if (words[0] == "set") {
		int y = stoi(words[1]);
		int x = stoi(words[2]);
		int val = stoi(words[3]);
		if (simType == 0) {
			simBool->setCell(y, x, val);
		}
		else if (simType == 1) {
			simInt->setCell(y, x, val);
		}
		else if (simType == 2) {
			simLong->setCell(y, x, val);
		}
		else if (simType == 3) {
			simLongLong->setCell(y, x, val);
		}
	}
	else if (words[0] == "exit") {
		cout << "\nI'll be back...\n" << endl;
		running = false;
	}
}




int main()
{
	srand(time(nullptr)); // use current time as seed for random generator
	cout << "Welcome to the Cellular Automata Simulator Program!" << endl;
	cout << "type 'help' to see the available commands" << endl;
	string line;
	while (running) {
		getline(cin, line);
		handleInput(line);
	}
	cout << "\n\n|_______ENDING PROGRAM________|\n";
	return 0;
}
