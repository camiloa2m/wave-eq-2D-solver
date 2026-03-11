#include "WaveEquation.hpp"
#include <cmath>

void WaveEquation::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing the mesh" << std::endl;

  {
    Triangulation<dim> mesh_serial;
    const double left = 0.0;
    const double right = 1.0;
    const unsigned int n_subdivisions = 50;
    
    if (dim == 1) {
      GridGenerator::subdivided_hyper_cube(mesh_serial, n_subdivisions, left, right, true);
    } else {
      // For simplicial meshes the 'colorize' option is not implemented in
      // deal.II's GridGenerator. Pass 'false' to avoid the ExcNotImplemented
      // exception raised when colorize==true.
      GridGenerator::subdivided_hyper_cube_with_simplices(mesh_serial, n_subdivisions, left, right, false);
    }

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);
    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Initializing the finite element space" << std::endl;

  fe = std::make_unique<FE_SimplexP<dim>>(r);
  quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Initializing the DoF handler" << std::endl;

  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);
  pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Initializing the linear system and vectors" << std::endl;

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Use of constraints to manage Dirichlet boundary conditions efficiently, 
  // deal.II v9.6: Streamlined constraints.
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // Dirichlet Condition: u = g (Defined here strongly)
  Functions::ZeroFunction<dim> g; // Homogeneous Dirichlet BCs.
  // WaveEquation::FunctionG g; // Function<dim> providing boundary values
  VectorTools::interpolate_boundary_values(dof_handler,
                                            0, // Boundary ID
                                            g, // value g
                                            constraints);
  constraints.close();
  // These constraints are automatically applied to our matrices and RHS during 
  // the distribute_local_to_global call in our assembly routine.

  TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints, false);
  sparsity.compress();

  mass_matrix.reinit(sparsity);
  stiffness_matrix.reinit(sparsity);
  system_matrix.reinit(sparsity);
  
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  // Initialize all kinematic vectors
  solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  
  old_solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  
  old_velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  old_velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  // Reuse temporary vectors for assembly and solving across timesteps to avoid reallocations.
  rhs_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  tmp_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  force_terms.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  // Set initial conditions by interpolating the FunctionU0 and FunctionV0 into our FE space.
  WaveEquation::FunctionU0 u_0;
  WaveEquation::FunctionV0 v_0;
  
  // Using projection instead of interpolation for better accuracy, especially for higher-order elements.
  // It ensures the initial energy in the FE space matches the continuous initial energy more closely.
  // Using QGaussSimplex with r+2 points for better accuracy in the projection.
  VectorTools::project(dof_handler, constraints, QGaussSimplex<dim>(r + 2), u_0, old_solution_owned);
  VectorTools::project(dof_handler, constraints, QGaussSimplex<dim>(r + 2), v_0, old_velocity_owned);
  // VectorTools::interpolate(dof_handler, u_0, old_solution_owned);
  // VectorTools::interpolate(dof_handler, v_0, old_velocity_owned);

  old_solution = old_solution_owned;
  old_velocity = old_velocity_owned;
}

void WaveEquation::assemble_matrices() 
{
  pcout << "Assembling matrices..." << std::endl;
  mass_matrix = 0;
  stiffness_matrix = 0;

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) continue;
    fe_values.reinit(cell);
    cell_mass = 0; 
    cell_stiffness = 0;
    for (unsigned int q = 0; q < n_q; ++q) {
      const double rho_val = rho(fe_values.quadrature_point(q));
      const double c_val = c(fe_values.quadrature_point(q));
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_mass(i, j) += rho_val * fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
          cell_stiffness(i, j) += c_val * c_val * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }
      }
    }
    // Neumann BCs if needed can be added here.
    // Current implementation assumes homogeneous Neumann (natural) BCs, so no additional terms are added.

    cell->get_dof_indices(dof_indices);
    constraints.distribute_local_to_global(cell_mass, dof_indices, mass_matrix);
    constraints.distribute_local_to_global(cell_stiffness, dof_indices, stiffness_matrix);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // system_matrix for the first-order coupled theta-method: M + (theta*dt)^2 * K 
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(theta * theta * delta_t * delta_t, stiffness_matrix);
}

void WaveEquation::solve_timestep() {
  // locally reused member vectors: rhs_owned, tmp_owned, force_terms

  // 1. Assemble the Force Vector (f) if this is time-dependent.
  // For our current problem, f is zero, but we include this step for generality and to show how to handle time-dependent forcing.
  // Reset force_terms before assembly to avoid accumulation across timesteps.
  
  force_terms = 0; 
  LambdaFunction ff(f); // Wrap the forcing lambda function into a dealii::Function to use with VectorTools
  ff.set_time(time + (1.0 - theta) * delta_t); 
  VectorTools::create_right_hand_side(dof_handler, *quadrature, ff, force_terms);

  // 2. Build RHS for Velocity
  // RHS = M*V_n - dt*K*(U_n + theta*(1-theta)*dt*V_n) + dt*F

  mass_matrix.vmult(rhs_owned, old_velocity_owned); 

  tmp_owned = old_solution_owned;
  tmp_owned.add(theta * (1.0 - theta) * delta_t, old_velocity_owned);

  TrilinosWrappers::MPI::Vector k_term(solution_owned.locally_owned_elements(), MPI_COMM_WORLD);
  stiffness_matrix.vmult(k_term, tmp_owned);
  
  rhs_owned.add(-delta_t, k_term);
  rhs_owned.add(delta_t, force_terms); // Add the forcing contribution

  // 3. Solve for V_{n+1}

  TrilinosWrappers::PreconditionSSOR prec;
  prec.initialize(system_matrix);
  SolverControl solver_control(1000, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

  cg.solve(system_matrix, velocity_owned, rhs_owned, prec);
  constraints.distribute(velocity_owned);
  pcout << "  Linear solver: " << solver_control.last_step() << " CG iterations." << std::endl;

  // 4. Update displacement U_{n+1} using the theta method: U_{n+1} = U_n + dt*(theta*V_{n+1} + (1-theta)*V_n)

  solution_owned = old_solution_owned;
  solution_owned.add(theta * delta_t, velocity_owned);
  solution_owned.add((1.0 - theta) * delta_t, old_velocity_owned);
  constraints.distribute(solution_owned);

  // Final sync and state update
  solution = solution_owned;
  velocity = velocity_owned;
  old_solution_owned = solution_owned;
  old_velocity_owned = velocity_owned;
  old_solution = solution_owned;
  old_velocity = velocity_owned;
}

void WaveEquation::compute_energy() const 
{
  // Energy = 0.5 * V^T * M * V + 0.5 * U^T * K * U
  TrilinosWrappers::MPI::Vector tmp(old_solution_owned.locally_owned_elements(), MPI_COMM_WORLD);
  
  mass_matrix.vmult(tmp, old_velocity_owned);
  double kinetic = 0.5 * (old_velocity_owned * tmp);
  
  stiffness_matrix.vmult(tmp, old_solution_owned);
  double potential = 0.5 * (old_solution_owned * tmp);
  
  pcout << "  Energy: " << kinetic + potential << std::endl;
}

void WaveEquation::output() const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "displacement");
  data_out.add_data_vector(dof_handler, velocity, "velocity");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record("./", output_file_name, timestep_number, MPI_COMM_WORLD);
}

void WaveEquation::run()
{
  setup();
  // M and K are constant in time for this problem, so we assemble them once.
  // If the problem had time-dependent coefficients or nonlinear terms appear, 
  // we would need to reassemblethese matrices at each timestep.
  // 
  assemble_matrices();

  output(); // Output initial state (t=0)
  compute_energy();

  while (time < T - 0.5 * delta_t) {
    time += delta_t;
    ++timestep_number;

    pcout << "Timestep " << timestep_number << " at t = " << time << std::endl;
    solve_timestep();
    compute_energy();
    
    // Output periodically to save disk space if delta_t is very small
    if (timestep_number % 10 == 0) {
      output();
    }
  }
}