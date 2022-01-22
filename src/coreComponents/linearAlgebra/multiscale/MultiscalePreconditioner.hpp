/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2019 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2019 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2019 Total, S.A
 * Copyright (c) 2019-     GEOSX Contributors
 * All right reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file MultiscalePreconditioner.hpp
 */

#ifndef GEOSX_LINEARALGEBRA_MULTISCALE_MULTISCALEPRECONDITIONER_HPP_
#define GEOSX_LINEARALGEBRA_MULTISCALE_MULTISCALEPRECONDITIONER_HPP_

#include "linearAlgebra/common/PreconditionerBase.hpp"
#include "linearAlgebra/utilities/LinearSolverParameters.hpp"
#include "mesh/MeshLevel.hpp"

#include <memory>

namespace geosx
{

namespace multiscale
{
template< typename LAI >
class LevelBuilderBase;
}

template< typename LAI >
class MultiscalePreconditioner : public PreconditionerBase< LAI >
{
public:

  /// Alias for base type
  using Base = PreconditionerBase< LAI >;

  /// Alias for vector type
  using Vector = typename Base::Vector;

  /// Alias for matrix type
  using Matrix = typename Base::Matrix;

  /// Alias for operator type
  using Operator = LinearOperator< Vector >;

  MultiscalePreconditioner( LinearSolverParameters params, MeshLevel & mesh );

  ~MultiscalePreconditioner();

  virtual void setup( Matrix const & mat ) override;

  virtual void apply( Vector const & src, Vector & dst ) const override;

private:

  void createLevels( Matrix const & mat,
                     DofManager const & dofManager );

  void printLevelInfo() const;

  struct Level
  {
    std::unique_ptr< multiscale::LevelBuilderBase< LAI > > builder;
    std::unique_ptr< PreconditionerBase< LAI > > presmoother;
    std::unique_ptr< PreconditionerBase< LAI > > postsmoother;
    Matrix const * matrix; ///< level matrix (operator)
    mutable Vector rhs;    ///< level right-hand side vector
    mutable Vector sol;    ///< level solution
    mutable Vector tmp;    ///< level temporary vector used to hold intermediate solutions
  };

  LinearSolverParameters m_params;

  MeshLevel & m_mesh;

  std::vector< Level > m_levels;

  std::unique_ptr< PreconditionerBase< LAI > > m_coarse_solver;

  bool m_initialized{ false };
};

} // namespace geosx

#endif //GEOSX_LINEARALGEBRA_MULTISCALE_MULTISCALEPRECONDITIONER_HPP_
