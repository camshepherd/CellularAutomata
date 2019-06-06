#include <chrono>

// Stopwatch is a very simple class to function as a stopwatch for the purposes of timing 
// code.
class Stopwatch
{
private:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_t> start_time;

public:
	Stopwatch() : start_time(clock_t::now())
	{
	}

	void reset()
	{
		start_time = clock_t::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<second_t>(clock_t::now() - start_time).count();
	}
};