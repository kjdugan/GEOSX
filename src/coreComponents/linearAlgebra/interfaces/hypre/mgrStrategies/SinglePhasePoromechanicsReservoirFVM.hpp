/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2020 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2020 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2020 TotalEnergies
 * Copyright (c) 2019-     GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file SinglePhasePoromechanics.hpp
 */

#ifndef GEOSX_LINEARALGEBRA_INTERFACES_HYPREMGRSINGLEPHASEPOROMECHANICSRESERVOIRFVM_HPP_
#define GEOSX_LINEARALGEBRA_INTERFACES_HYPREMGRSINGLEPHASEPOROMECHANICSRESERVOIRFVM_HPP_

#include "linearAlgebra/interfaces/hypre/HypreMGR.hpp"

namespace geosx
{

namespace hypre
{

namespace mgr
{

/**
 * @brief SinglePhasePoromechanicsReservoirFVM strategy.
 *
 * dofLabel: 0 = displacement, x-component
 * dofLabel: 1 = displacement, y-component
 * dofLabel: 2 = displacement, z-component
 * dofLabel: 3 = pressure
 * dofLabel: 4 = well pressure
 * dofLabel: 5 = well rate
 *
 * 2-level MGR reduction strategy
 *   - 1st level: eliminate displacements (0,1,2)
 *   - 2nd level: eliminate the well block
 *   - The coarse grid is solved with BoomerAMG.
 *
 */
class SinglePhasePoromechanicsReservoirFVM : public MGRStrategyBase< 2 >
{
public:

  /**
   * @brief Constructor.
   */
  explicit SinglePhasePoromechanicsReservoirFVM( arrayView1d< int const > const & )
    : MGRStrategyBase( 6 )
  {
    // Level 0: eliminate displacements
    m_labels[0].push_back( 3 );
    m_labels[0].push_back( 4 );
    m_labels[0].push_back( 5 );
    // Level 1: eliminate the well block
    m_labels[1].push_back( 3 );

    setupLabels();

    // Level 0
    m_levelFRelaxMethod[0]     = MGRFRelaxationMethod::amgVCycle;
    m_levelInterpType[0]       = MGRInterpolationType::jacobi;
    m_levelRestrictType[0]     = MGRRestrictionType::injection;
    m_levelCoarseGridMethod[0] = MGRCoarseGridMethod::nonGalerkin;

    // Level 1
    m_levelFRelaxMethod[1]     = MGRFRelaxationMethod::singleLevel;
    m_levelInterpType[1]       = MGRInterpolationType::blockJacobi;
    m_levelRestrictType[1]     = MGRRestrictionType::injection;
    m_levelCoarseGridMethod[1] = MGRCoarseGridMethod::galerkin;

    m_numGlobalSmoothSweeps = 0;
  }

  /**
   * @brief Setup the MGR strategy.
   * @param precond preconditioner wrapper
   * @param mgrData auxiliary MGR data
   */
  void setup( LinearSolverParameters::MGR const &,
              HyprePrecWrapper & precond,
              HypreMGRData & mgrData )
  {
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetCpointsByPointMarkerArray( precond.ptr,
                                                                  m_numBlocks, numLevels,
                                                                  m_numLabels, m_ptrLabels,
                                                                  mgrData.pointMarkers.data() ) );

    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetLevelFRelaxMethod( precond.ptr, toUnderlyingPtr( m_levelFRelaxMethod ) ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetLevelInterpType( precond.ptr, toUnderlyingPtr( m_levelInterpType ) ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetLevelRestrictType( precond.ptr, toUnderlyingPtr( m_levelRestrictType ) ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetCoarseGridMethod( precond.ptr, toUnderlyingPtr( m_levelCoarseGridMethod ) ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetNonCpointsToFpoints( precond.ptr, 1 ));
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetPMaxElmts( precond.ptr, 0 ));
    GEOSX_LAI_CHECK_ERROR( HYPRE_MGRSetMaxGlobalSmoothIters( precond.ptr, m_numGlobalSmoothSweeps ) );

    GEOSX_LAI_CHECK_ERROR( HYPRE_BoomerAMGCreate( &mgrData.coarseSolver.ptr ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_BoomerAMGSetPrintLevel( mgrData.coarseSolver.ptr, 0 ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_BoomerAMGSetMaxIter( mgrData.coarseSolver.ptr, 1 ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_BoomerAMGSetTol( mgrData.coarseSolver.ptr, 0.0 ) );
    GEOSX_LAI_CHECK_ERROR( HYPRE_BoomerAMGSetRelaxOrder( mgrData.coarseSolver.ptr, 1 ) );

    mgrData.coarseSolver.setup = HYPRE_BoomerAMGSetup;
    mgrData.coarseSolver.solve = HYPRE_BoomerAMGSolve;
    mgrData.coarseSolver.destroy = HYPRE_BoomerAMGDestroy;

    // Configure the BoomerAMG solver used as F-relaxation for the first level
    setMechanicsFSolver( precond, mgrData );
  }
};

} // namespace mgr

} // namespace hypre

} // namespace geosx

#endif /*GEOSX_LINEARALGEBRA_INTERFACES_HYPREMGRSINGLEPHASEPOROMECHANICSRESERVOIRFVM_HPP_*/
