#include "WaveEquation.hpp"

// Main function.
int main(int argc, char *argv[])
{
  // Initialize MPI for parallel execution
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  constexpr unsigned int dim = WaveEquation::dim;

  // Density of the medium (rho)
  // For a homogeneous medium, we keep this constant.
  const auto rho = [](const Point<dim> & /*p*/) { return 1.0; }; 

  // Wave speed (c)
  // Represents the stiffness characteristic of the medium. 
  const auto c = [](const Point<dim> & /*p*/) { return 1.0; }; 

  // External forcing function (f)
  // Set to 0.0 because our wave is driven by the initial displacement (the Gaussian pulse) 
  // defined in the FunctionU0 class in WaveEquation.hpp.
  const auto f = [](const Point<dim> & /*p*/, const double & /*t*/) {
    return 0.0;
  };

  // Theta parameter for the time-stepping scheme (theta method).
  // Crank-Nicolson (theta=0.5) : Unconditionally stable and second-order accurate in time.
  // Backward Euler (theta=1.0) : Unconditionally stable and first-order accurate in time.
  // Forward Euler (theta=0.0) : Conditionally stable and first-order accurate in time.
  const double theta  = 0.5; 


  // Instantiate the WaveEquation problem
  WaveEquation problem(
      /* mesh_file_name = */ "wave_domain.msh", // Used for output naming since we generate the mesh internally
      /* degree         = */ 1,                 // Polynomial degree (p=1 for linear elements)
      /* T              = */ 2.0,               // Final time. T=2.0 is enough to see the wave hit boundaries and reflect
      /* delta_t        = */ 0.005,             // Time step. 
      /* theta          = */ theta,             // Theta parameter for the time-stepping scheme (theta method)
      rho,
      c,
      f
  );

  // Execute the simulation
  problem.run();

  return 0;
}

/*
EXPECTED BEHAVIOR:
- At t=0, the energy is purely potential due to the initial Gaussian displacement.
- As time advances, the pulse will expand outward circularly (in 2D).
- The total energy E_n printed to the console should remain constant throughout the simulation.
- Because we are using standard Dirichlet boundaries (fixed ends where u=0) by default in the .cpp file, 
  the wave will reflect perfectly off the edges of the domain, bouncing back inward.
*/