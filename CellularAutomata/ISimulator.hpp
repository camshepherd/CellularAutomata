#pragma once
#include "Stopwatch.hpp"
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace CellularAutomata {
	/** Interface defining the behaviour of any class to simulate synchronous cellular automata models
	*/
	template <typename T>
	class ISimulator {
	protected:
		Stopwatch timer;
		double elapsedTime = 0;
	public:
		/** Constructor 1. Default constructor
		*/
		ISimulator() { elapsedTime = 0; };

		/** Generate the next frame of the simulation
		@param steps: Number of steps to go forward
		*/
		virtual double stepForward(int steps = 1) = 0;

		/**	Simulate the model to additional frames for a set time period
		@param seconds: Number of seconds to simulate for
		*/
		double stepForwardTime(double seconds);

		/** Create a new frame in the simulation that is completely blank
		*/
		virtual bool blankFrame() = 0;

		/** Create a new frame in the simulation that is an exact copy of the most recent
		*/
		virtual bool copyFrame() = 0;

		/** Get the value of the target cell in the cellStore
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		@param t: The timestep of the frame to read the data from. A value of -1 uses the most-recent
		*/
		virtual T getCell(int y, int x, int t = -1) const = 0;

		/** Get the number of frames that exist in the simulation
		*/
		virtual int getNumFrames() const = 0;

		/** Get the size of the simulation in the y-direction
		*/
		virtual int getYDim() = 0;

		/** Get the size of the simulation in the x-direction
		*/
		virtual int getXDim() = 0;

		/** Get the maximum state representation value that can be used in the current model
		*/
		virtual T getMaxValidState() = 0;

		/** Set the list of the parameters needed by the simulator
		 */
		virtual bool setDimensions(int y, int x) = 0;

		/** Set the value of a specific cell
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		@param new_val: The new value to be assigned to the cell
		@param t: The timestep of the frame that is to be altered. A value of -1 uses the most recent
		*/
		virtual bool setCell(int y, int x, T new_val, int t = -1) = 0;

		/** Wipe all data from the stored simulation
		@param addBlankFirstFrame: Whether to leave the cellStore empty or for it to have an empty first frame. Default is true: add a blank frame
		*/
		virtual bool clear(bool addBlankFirstFrame = true) = 0;

		/** Write all frames simulated to file
		@param filename: The path of the file that is to be written to
		*/
		bool writeData(std::string filename);

		/** Evaluate whether the two simulations are equal. This is the case if the dimensions are the same and all 
		 * frames have been simulated in the same way in both simulators.
		 */
		bool friend operator==(ISimulator<T>& a, ISimulator<T>& b) {
			if (a.getXDim() != b.getXDim()) {
				return false;
			}
			else if (a.getYDim() != b.getYDim()) {
				return false;
			}
			else {
				for (int y = 0; y < a.getYDim(); ++y) {
					for (int x = 0; x < a.getXDim(); ++x) {
						for (int t = 0; t < std::min<int>(a.getNumFrames(), b.getNumFrames()); ++t) {
							if (a.getCell(y, x, t) != b.getCell(y, x, t)) {
								return false;
							}
						}

					}
				}
				return true;
			}
		}

		/** Print the specified frame to stdout
		@param frameNum: The frame number to print; defaults to -1: highest-numbered
		*/
		virtual void printFrame(int frameNum=-1) = 0;
	};
}
#include "ISimulator.inl"