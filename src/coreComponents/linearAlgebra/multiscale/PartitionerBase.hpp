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
 * @file PartitionerBase.hpp
 */

#ifndef GEOSX_LINEARALGEBRA_MULTISCALE_PARTITIONERBASE_HPP
#define GEOSX_LINEARALGEBRA_MULTISCALE_PARTITIONERBASE_HPP

#include "linearAlgebra/multiscale/MeshLevel.hpp"
#include "linearAlgebra/utilities/LinearSolverParameters.hpp"

#include <memory>

namespace geosx
{
namespace multiscale
{

class PartitionerBase
{
public:

  static std::unique_ptr< PartitionerBase >
  create( LinearSolverParameters::Multiscale::Coarsening params );

  virtual ~PartitionerBase() = default;

  /**
   * @brief Generate a partitioning of fine-scale mesh cells.
   * @param mesh the fine-scale mesh
   * @param partition the partition index output array (that must be properly sized)
   * @return the number of partitions generated
   */
  virtual localIndex generate( multiscale::MeshLevel const & mesh,
                               arrayView1d< localIndex > const & partition ) = 0;

  /**
   * @brief Store auxiliary partitioning-related data on the coarse mesh.
   * @param fineMesh the fine mesh
   * @param coarseMesh the coarse mesh
   * @param partition partition index array produced in generate()
   *
   * This function can be used to
   */
  virtual void setCoarseData( multiscale::MeshLevel & coarseMesh ) const
  {
    GEOSX_UNUSED_VAR( coarseMesh );
  };
};

} // namespace multiscale
} // namespace geosx

#endif //GEOSX_LINEARALGEBRA_MULTISCALE_PARTITIONERBASE_HPP
