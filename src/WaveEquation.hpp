#ifndef WAVE_EQUATION_HPP
#define WAVE_EQUATION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the second-order wave equation problem.
 */
class WaveEquation
{
public:
  // Physical dimension (1D, 2D, 3D). Set to 2 for a 2D wave problem.
  static constexpr unsigned int dim = 2;

  // Initial displacement condition: u(x, 0) = u_0(x)
  class FunctionU0 : public Function<dim>
  {
  public:
    FunctionU0() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // Example: A Gaussian pulse centered at (0.5, 0.5)
      const double d2 = (p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5);
      return std::exp(-100.0 * d2); 
    }
  };

  // Initial velocity condition: \dot{u}(x, 0) = v_0(x)
  class FunctionV0 : public Function<dim>
  {
  public:
    FunctionV0() = default;

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0; // Starts from rest
    }
  };

  // Wrap a lambda into a dealii::Function
  class LambdaFunction : public Function<dim>
  {
  public:
    std::function<double(const Point<dim>&, double)> fn;
    LambdaFunction(std::function<double(const Point<dim>&, double)> f)
      : Function<dim>(), fn(std::move(f)) {}
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return fn(p, this->get_time());
    }
  };


  // Constructor
  WaveEquation(const std::string                                       &mesh_file_name_,
               const unsigned int                                      &r_,
               const double                                            &T_,
               const double                                            &delta_t_,
               const double                                            &theta_,
               const std::function<double(const Point<dim> &)>         &rho_,
               const std::function<double(const Point<dim> &)>         &c_,
               const std::function<double(const Point<dim> &, const double &)> &f_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , T(T_)
    , delta_t(delta_t_)
    , theta(theta_)
    , rho(rho_)
    , c(c_)
    , f(f_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh(MPI_COMM_WORLD)
  {}

  // Run the time-dependent simulation.
  void
  run();

protected:
  // Initialization of mesh, FE space, DoFs, and linear algebra objects.
  void
  setup();

  // Builds M and K once
  void
  assemble_matrices();

  // Computes RHS and solves for U_{n+1}
  void
  solve_timestep();

  // Computes and prints the total mechanical energy E_n.
  void
  compute_energy() const;

  // Output to VTU/PVTU.
  void
  output() const;

  // --- Parameters ---
  const std::string mesh_file_name;
  const unsigned int r;
  const double T;
  const double delta_t;
  const double theta;

  double time = 0.0;
  unsigned int timestep_number = 0;

  // Physical properties
  std::function<double(const Point<dim> &)> rho;
  std::function<double(const Point<dim> &)> c;
  std::function<double(const Point<dim> &, const double &)> f;

  // MPI and Output
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  // FE structure
  parallel::fullydistributed::Triangulation<dim> mesh;
  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;
  DoFHandler<dim> dof_handler;

  // Linear Algebra Objects
  AffineConstraints<double> constraints;           // For managing Dirichlet BCs and hanging nodes
  TrilinosWrappers::SparseMatrix mass_matrix;      // M
  TrilinosWrappers::SparseMatrix stiffness_matrix; // K
  TrilinosWrappers::SparseMatrix system_matrix;    // M + (dt^2/4)*K
  TrilinosWrappers::MPI::Vector system_rhs;        // RHS for the linear system

  // Kinematic state vectors (owned for solving, full for evaluation/output)
  TrilinosWrappers::MPI::Vector solution_owned, solution;           // U_{n+1}
  TrilinosWrappers::MPI::Vector old_solution_owned, old_solution;   // U_n
  TrilinosWrappers::MPI::Vector velocity_owned, velocity;           // V_{n+1}
  TrilinosWrappers::MPI::Vector old_velocity_owned, old_velocity;   // V_n

private:
  // Temporary vectors for assembly and solving
  TrilinosWrappers::MPI::Vector rhs_owned;
  TrilinosWrappers::MPI::Vector tmp_owned;
  TrilinosWrappers::MPI::Vector force_terms;
};

#endif