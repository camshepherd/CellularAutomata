#pragma once
#include "ISimulator.hpp"
#include "IRules.hpp"
#include <vector>

namespace CellularAutomata {
	/** Abstract class that implements all necessary generic functions for a simulator based on std::vector
	*/
	template <typename T>
	class SimulatorVector :
		public ISimulator<T>
	{
	protected:
		// The store of all frames in all timesteps
		std::vector<std::vector<std::vector<T>>> cellStore;
		int y_dim, x_dim;

		const IRules<T>& rules;
	public:
		/** Constructor 1. Construct the simulator from the given dependencies
		@param y: Size of the simulation in the y direction
		@param x: Size of the simulation in the x direction
		@param rules: The rules to govern the simulation
		*/
		SimulatorVector(const int y, const int x, const IRules<T>& rules);


		/** Destructor 1. Default destructor
		*/
		~SimulatorVector();

		/** Reset the simulation and wipe all simulated frames
		@param addBlankFirstFrame: Whether to make the cellStore empty(false), or to add an initial blank frame(true, default)
		*/
		virtual bool clear(bool addBlankFirstFrame = true) override;

		/** Get the number of frames that have been simulated so far
		*/
		virtual int getNumFrames() const override;

		/** Set the state of the given cell
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		@param new_val: The value to assign to the given cell
		@param t: The frame to alter. Default (-1) targets the most recent
		*/
		virtual bool setCell(int y, int x, T new_val, int t = -1) override;

		/** Create a new blank frame in the simulation
		*/
		virtual bool blankFrame() override;

		/** Get the value of the specified cell
		@param y: The y-coordinate of the target cell
		@param x: The x-coordinate of the target cell
		@param t: The frame to target
		*/
		virtual T getCell(int y, int x, int t = -1) const override;

		/** Get the size of the simulation in the y dimension
		*/
		int getYDim() override;

		/** Get the size of the simulation in the x dimension
		*/
		int getXDim() override;

		/** Insert a new frame in the simulation, exactly duplicating the back frame
		*/
		bool copyFrame() override;
	};
}
#include "SimulatorVector.inl"