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
 * @file MsrsbLevelBuilder.cpp
 */

#include "MsrsbLevelBuilder.hpp"

#include "linearAlgebra/interfaces/InterfaceTypes.hpp"
#include "linearAlgebra/multiscale/MeshData.hpp"
#include "linearAlgebra/multiscale/MeshUtils.hpp"
#include "linearAlgebra/multiscale/MsrsbUtils.hpp"
#include "linearAlgebra/utilities/TransposeOperator.hpp"
#include "mesh/DomainPartition.hpp"
#include "mesh/mpiCommunications/CommunicationTools.hpp"

namespace geosx
{
namespace multiscale
{

template< typename LAI >
MsrsbLevelBuilder< LAI >::MsrsbLevelBuilder( string name,
                                             LinearSolverParameters::Multiscale params )
  : LevelBuilderBase< LAI >( std::move( name ), std::move( params ) ),
  m_mesh( m_name )
{}

namespace msrsb
{
namespace node
{

ArrayOfSets< localIndex >
buildFineSubdomainMap( MeshObjectManager const & fineNodeManager,
                       MeshObjectManager const & fineCellManager,
                       arrayView1d< string const > const & boundaryNodeSets )
{
  MeshObjectManager::MapViewConst const nodeToCell = fineNodeManager.toDualRelation().toViewConst();

  // count the row lengths
  array1d< localIndex > rowCounts( fineNodeManager.size() );
  forAll< parallelHostPolicy >( fineNodeManager.size(), [=, rowCounts = rowCounts.toView()]( localIndex const inf )
  {
    rowCounts[inf] = nodeToCell.sizeOfSet( inf );
  } );
  for( string const & setName: boundaryNodeSets )
  {
    SortedArrayView< localIndex const > const set = fineNodeManager.getSet( setName ).toViewConst();
    forAll< parallelHostPolicy >( set.size(), [=, rowCounts = rowCounts.toView()]( localIndex const i )
    {
      ++rowCounts[set[i]];
    } );
  }

  // Resize from row lengths
  ArrayOfSets< localIndex > nodeToSubdomain;
  nodeToSubdomain.resizeFromCapacities< parallelHostPolicy >( rowCounts.size(), rowCounts.data() );

  // Fill the map
  localIndex numBoundaries = 0;
  for( string const & setName: boundaryNodeSets )
  {
    ++numBoundaries;
    SortedArrayView< localIndex const > const set = fineNodeManager.getSet( setName ).toViewConst();
    forAll< parallelHostPolicy >( set.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inf )
    {
      nodeToSub.insertIntoSet( set[inf], -numBoundaries ); // use negative indices to differentiate boundary subdomains
    } );
  }

  arrayView1d< localIndex const > const coarseCellLocalIndex = fineCellManager.getExtrinsicData< meshData::CoarseCellLocalIndex >();
  forAll< parallelHostPolicy >( fineNodeManager.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inf )
  {
    for( localIndex const icf: nodeToCell[inf] )
    {
      nodeToSub.insertIntoSet( inf, coarseCellLocalIndex[icf] );
    }
  } );

  return nodeToSubdomain;
}

ArrayOfSets< localIndex >
buildCoarseSubdomainMap( MeshObjectManager const & coarseNodeManager,
                         arrayView1d< string const > const & boundaryNodeSets )
{
  MeshObjectManager::MapViewConst const nodeToCell = coarseNodeManager.toDualRelation().toViewConst();

  // count the row lengths
  array1d< localIndex > rowCounts( coarseNodeManager.size() );
  forAll< parallelHostPolicy >( coarseNodeManager.size(), [=, rowCounts = rowCounts.toView()]( localIndex const inc )
  {
    rowCounts[inc] = nodeToCell.sizeOfSet( inc );
  } );
  for( string const & setName: boundaryNodeSets )
  {
    SortedArrayView< localIndex const > const set = coarseNodeManager.getSet( setName ).toViewConst();
    forAll< parallelHostPolicy >( set.size(), [=, rowCounts = rowCounts.toView()]( localIndex const i )
    {
      ++rowCounts[set[i]];
    } );
  }

  // Resize from row lengths
  ArrayOfSets< localIndex > nodeToSubdomain;
  nodeToSubdomain.resizeFromCapacities< parallelHostPolicy >( rowCounts.size(), rowCounts.data() );

  // Fill the map
  localIndex numBoundaries = 0;
  for( string const & setName: boundaryNodeSets )
  {
    ++numBoundaries;
    SortedArrayView< localIndex const > const set = coarseNodeManager.getSet( setName ).toViewConst();
    forAll< parallelHostPolicy >( set.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inc )
    {
      nodeToSub.insertIntoSet( set[inc], -numBoundaries ); // use negative indices to differentiate boundary subdomains
    } );
  }
  forAll< parallelHostPolicy >( coarseNodeManager.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inc )
  {
    arraySlice1d< localIndex const > const ccells = nodeToCell[inc];
    nodeToSub.insertIntoSet( inc, ccells.begin(), ccells.end() );
  } );

  return nodeToSubdomain;
}

/**
 * @brief Build the basic sparsity pattern for prolongation.
 *
 * Support of a coarse nodal basis function is defined as the set of fine-scale nodes
 * that are adjacent exclusively to subdomains (coarse cells or boundaries) that are
 * also adjacent to that coarse node.
 */
ArrayOfSets< localIndex >
buildSupports( multiscale::MeshLevel const & fine,
               multiscale::MeshLevel const & coarse,
               arrayView1d< string const > const & boundaryNodeSets,
               arrayView1d< integer > const & supportBoundaryIndicator )
{
  GEOSX_MARK_FUNCTION;

  ArrayOfSets< localIndex > const fineNodeToSubdomain =
    buildFineSubdomainMap( fine.nodeManager(), fine.cellManager(), boundaryNodeSets );
  ArrayOfSets< localIndex > const coarseNodeToSubdomain =
    buildCoarseSubdomainMap( coarse.nodeManager(), boundaryNodeSets );

  ArrayOfSetsView< localIndex const > const coarseCellToNode = coarse.cellManager().toDualRelation().toViewConst();
  arrayView1d< localIndex const > const coarseNodeIndex = fine.nodeManager().getExtrinsicData< meshData::CoarseNodeLocalIndex >();

  // Algorithm:
  // Loop over all fine nodes.
  // If node is a coarse node, immediately assign to its own support.
  // Otherwise, get a list of adjacent coarse cells.
  // If list is length 1, assign the node to supports of all coarse nodes adjacent to that coarse cell.
  // Otherwise, collect a unique list of candidate coarse nodes by visiting them through coarse cells.
  // For each candidate, check that fine node's subdomain list is included in the candidates subdomain list.
  // Otherwise, discard the candidate.
  //
  // All above is done twice: once to count (or get upper bound on) row lengths, once to actually build supports.
  // For the last case, don't need to check inclusion when counting, just use number of candidates as upper bound.

  // Count row lengths and fill boundary indicators
  array1d< localIndex > rowLengths( fine.nodeManager().size() );
  forAll< parallelHostPolicy >( fine.nodeManager().size(), [coarseNodeIndex, coarseCellToNode,
    rowLengths = rowLengths.toView(),
    fineNodeToSubdomain = fineNodeToSubdomain.toViewConst(),
    supportBoundaryIndicator = supportBoundaryIndicator.toView()]( localIndex const inf )
  {
    if( coarseNodeIndex[inf] >= 0 )
    {
      rowLengths[inf] = 1;
      supportBoundaryIndicator[inf] = 1;
    }
    else if( fineNodeToSubdomain.sizeOfSet( inf ) == 1 )
    {
      rowLengths[inf] = coarseCellToNode.sizeOfSet( fineNodeToSubdomain( inf, 0 ) );
    }
    else
    {
      localIndex numCoarseNodes = 0;
      meshUtils::forUniqueNeighbors< 256 >( inf, fineNodeToSubdomain, coarseCellToNode, [&]( localIndex )
      {
        ++numCoarseNodes;
      } );
      rowLengths[inf] = numCoarseNodes;
      supportBoundaryIndicator[inf] = 1;
    }
  } );

  // Create and resize
  ArrayOfSets< localIndex > supports;
  supports.resizeFromCapacities< parallelHostPolicy >( rowLengths.size(), rowLengths.data() );

  // Fill the map
  forAll< parallelHostPolicy >( fine.nodeManager().size(), [coarseNodeIndex, coarseCellToNode,
    supports = supports.toView(),
    fineNodeToSubdomain = fineNodeToSubdomain.toViewConst(),
    coarseNodeToSubdomain = coarseNodeToSubdomain.toViewConst()]( localIndex const inf )
  {
    if( coarseNodeIndex[inf] >= 0 )
    {
      supports.insertIntoSet( inf, coarseNodeIndex[inf] );
    }
    else if( fineNodeToSubdomain.sizeOfSet( inf ) == 1 )
    {
      arraySlice1d< localIndex const > const coarseNodes = coarseCellToNode[fineNodeToSubdomain( inf, 0 )];
      supports.insertIntoSet( inf, coarseNodes.begin(), coarseNodes.end() );
    }
    else
    {
      arraySlice1d< localIndex const > const fsubs = fineNodeToSubdomain[inf];
      meshUtils::forUniqueNeighbors< 256 >( inf, fineNodeToSubdomain, coarseCellToNode, [&]( localIndex const inc )
      {
        arraySlice1d< localIndex const > const csubs = coarseNodeToSubdomain[inc];
        if( std::includes( csubs.begin(), csubs.end(), fsubs.begin(), fsubs.end() ) )
        {
          supports.insertIntoSet( inf, inc );
        }
      } );
    }
  } );

  return supports;
}

CRSMatrix< real64, globalIndex >
buildTentativeProlongation( multiscale::MeshLevel const & fineMesh,
                            multiscale::MeshLevel const & coarseMesh,
                            ArrayOfSetsView< localIndex const > const & supports,
                            integer const numComp )
{
  GEOSX_MARK_FUNCTION;

  // Build support regions and tentative prolongation
  ArrayOfSets< localIndex > const nodalConn = buildNodalConnectivity( fineMesh.nodeManager(), fineMesh.cellManager() );
  arrayView1d< localIndex const > const coarseNodes = coarseMesh.nodeManager().getExtrinsicData< meshData::FineNodeLocalIndex >().toViewConst();
  array1d< localIndex > const initPart = makeSeededPartition( nodalConn.toViewConst(), coarseNodes, supports );

  // Construct the tentative prolongation, consuming the sparsity pattern
  CRSMatrix< real64, globalIndex > localMatrix;
  {
    SparsityPattern< globalIndex > localPattern =
      buildProlongationSparsity( fineMesh.nodeManager(), coarseMesh.nodeManager(), supports, numComp );
    localMatrix.assimilate< parallelHostPolicy >( std::move( localPattern ) );
  }

  // Add initial unity values
  arrayView1d< globalIndex const > const coarseLocalToGlobal = coarseMesh.nodeManager().localToGlobalMap();
  forAll< parallelHostPolicy >( fineMesh.nodeManager().numOwnedObjects(),
                                [=, localMatrix = localMatrix.toViewConstSizes()]( localIndex const inf )
  {
    if( initPart[inf] >= 0 )
    {
      real64 const value = 1.0;
      for( integer ic = 0; ic < numComp; ++ic )
      {
        globalIndex const col = coarseLocalToGlobal[initPart[inf]] * numComp + ic;
        localMatrix.addToRow< serialAtomic >( inf * numComp + ic, &col, &value, 1 );
      }
    }
  } );

  return localMatrix;
}

} // namespace node

namespace cell
{

/**
 * @brief Build the basic sparsity pattern for prolongation.
 *
 * Support of a coarse nodal basis function is defined as the set of fine-scale nodes
 * that are adjacent exclusively to subdomains (coarse cells or boundaries) that are
 * also adjacent to that coarse node.
 */
ArrayOfSets< localIndex >
buildSupports( multiscale::MeshLevel & fine,
               multiscale::MeshLevel const & coarse,
               arrayView1d< string const > const & boundaryNodeSets,
               arrayView1d< integer > const & supportBoundaryIndicator )
{
  GEOSX_MARK_FUNCTION;

  // First, build the nodal partition to act as dual "volumes"
  // Unfortunately, we need nodal supports for that (in order to limit partition growth).
  array1d< localIndex > const nodalPartLocal = [&]
  {
    array1d< integer > const supportBoundaryIndicators( fine.nodeManager().size() );
    ArrayOfSets< localIndex > const nodalSupports = msrsb::node::buildSupports( fine,
                                                                                coarse,
                                                                                boundaryNodeSets,
                                                                                supportBoundaryIndicators );

    ArrayOfSets< localIndex > const nodalConn = buildNodalConnectivity( fine.nodeManager(), fine.cellManager() );
    arrayView1d< localIndex const > const coarseNodes = coarse.nodeManager().getExtrinsicData< meshData::FineNodeLocalIndex >().toViewConst();
    return makeSeededPartition( nodalConn.toViewConst(), coarseNodes, nodalSupports.toViewConst() );
  }();

  // Need to make nodal partition global and synced across ranks to have a consistent dual
  array1d< globalIndex > & nodalPart =
    fine.nodeManager().registerWrapper< meshData::NodalPartitionGlobalIndex::type >( meshData::NodalPartitionGlobalIndex::key() ).reference();
  {
    globalIndex const firstLocalNodeIndex = coarse.nodeManager().localToGlobalMap()[0];
    forAll< parallelHostPolicy >( fine.nodeManager().numOwnedObjects(),
                                  [firstLocalNodeIndex,
                                    nodalPart = nodalPart.toView(),
                                    nodalPartLocal = nodalPartLocal.toView()]( localIndex const inf )
    {
      nodalPart[inf] = nodalPartLocal[inf] + firstLocalNodeIndex;
    } );
    string_array fieldNames;
    fieldNames.emplace_back( meshData::NodalPartitionGlobalIndex::key() );
    CommunicationTools::getInstance().synchronizeFields( fieldNames, fine.nodeManager(), fine.domain()->getNeighbors(), false );
  }

  // TODO remove debugging output
  fine.writeNodeData( { meshData::NodalPartitionGlobalIndex::key() } );

  // Make an adjacency map of fine cells to nodal partitions
  ArrayOfSets< globalIndex > const cellToNodalPart =
    meshUtils::buildFineObjectToSubdomainMap( fine.cellManager(), nodalPart.toViewConst(), {} );

  array1d< localIndex > const dualVertexCells =
    meshUtils::findCoarseNodesByDualPartition( fine.cellManager().toDualRelation().toViewConst(),
                                               fine.nodeManager().toDualRelation().toViewConst(),
                                               cellToNodalPart.toViewConst(), 3 );

  // Construct adjacency maps between dual coarse "vertices" (cells) and volumes (nodal partitions)
  // TODO

  ArrayOfSets< localIndex > supports;
  return supports;
}

CRSMatrix< real64, globalIndex >
buildTentativeProlongation( multiscale::MeshLevel const & fineMesh,
                            multiscale::MeshLevel const & coarseMesh,
                            ArrayOfSetsView< localIndex const > const & supports,
                            integer const numComp )
{
  // TODO
  CRSMatrix< real64, globalIndex > prolongation;
  return prolongation;
}

} // namespace cell

} // namespace msrsb

template< typename LAI >
void MsrsbLevelBuilder< LAI >::initializeCoarseLevel( LevelBuilderBase< LAI > & fine_level )
{
  GEOSX_MARK_FUNCTION;

  MsrsbLevelBuilder< LAI > & fine = dynamicCast< MsrsbLevelBuilder< LAI > & >( fine_level );
  m_numComp = fine.m_numComp;
  m_location = fine.m_location;

  // Coarsen mesh
  m_mesh.buildCoarseMesh( fine.mesh(), m_params.coarsening, m_params.boundarySets );

  // Write data back to GEOSX for visualization and debug
  if( m_params.debugLevel >= 1 )
  {
    GEOSX_LOG_RANK_0( GEOSX_FMT( "[MsRSB] {}: generated coarse grid with {} global cells and {} global nodes",
                                 m_name,
                                 m_mesh.cellManager().maxGlobalIndex() + 1,
                                 m_mesh.nodeManager().maxGlobalIndex() + 1 ) );
    GEOSX_LOG_RANK( GEOSX_FMT( "[MsRSB] {}: generated coarse grid with {} local cells and {} local nodes",
                               m_name,
                               m_mesh.cellManager().numOwnedObjects(),
                               m_mesh.nodeManager().numOwnedObjects() ) );

    m_mesh.writeCellData( { ObjectManagerBase::viewKeyStruct::ghostRankString() } );
    m_mesh.writeNodeData( { meshData::FineNodeLocalIndex::key() } );
    fine.mesh().writeCellData( { meshData::CoarseCellLocalIndex::key(),
                                 meshData::CoarseCellGlobalIndex::key() } );
    fine.mesh().writeNodeData( { meshData::CoarseNodeLocalIndex::key(),
                                 meshData::CoarseNodeGlobalIndex::key() } );
  }

  MeshObjectManager const & coarseMgr = m_location == DofManager::Location::Node ? m_mesh.nodeManager() : m_mesh.cellManager();
  MeshObjectManager const & fineMgr = m_location == DofManager::Location::Node ? fine.mesh().nodeManager() : fine.mesh().cellManager();

  // Create a "fake" coarse matrix (no data, just correct sizes/comms)
  localIndex const numLocalDof = coarseMgr.numOwnedObjects() * m_numComp;
  m_matrix.createWithLocalSize( numLocalDof, numLocalDof, 0, fine.matrix().comm() );

  // Build initial (tentative) prolongation operator
  // Do this in a local scope so supports map does not outlive its usefulness
  CRSMatrix< real64, globalIndex > localProlongation;
  {
    array1d< integer > supportBoundaryIndicators( fineMgr.size() );

    if( m_location == DofManager::Location::Node )
    {
      ArrayOfSets< localIndex > const supports = msrsb::node::buildSupports( fine.mesh(),
                                                                             m_mesh,
                                                                             m_params.boundarySets,
                                                                             supportBoundaryIndicators );
      localProlongation = msrsb::node::buildTentativeProlongation( fine.mesh(),
                                                                   m_mesh,
                                                                   supports.toViewConst(),
                                                                   m_numComp );
    }
    else
    {
      ArrayOfSets< localIndex > const supports = msrsb::cell::buildSupports( fine.mesh(),
                                                                             m_mesh,
                                                                             m_params.boundarySets,
                                                                             supportBoundaryIndicators );
    }

    msrsb::makeGlobalDofLists( supportBoundaryIndicators,
                               m_numComp,
                               fineMgr.numOwnedObjects(),
                               fine.matrix().ilower(),
                               m_boundaryDof,
                               m_interiorDof );
  }

  m_prolongation.create( localProlongation.toViewConst(), coarseMgr.numOwnedObjects() * m_numComp, fine.matrix().comm() );
  m_restriction = msrsb::makeRestriction( m_params, m_prolongation );
}

template< typename LAI >
void MsrsbLevelBuilder< LAI >::initializeFineLevel( geosx::MeshLevel & mesh,
                                                    DofManager const & dofManager,
                                                    string const & fieldName,
                                                    MPI_Comm const & comm )
{
  GEOSX_MARK_FUNCTION;

  m_numComp = dofManager.numComponents( fieldName );
  m_location = dofManager.location( fieldName );
  m_mesh.buildFineMesh( mesh, dofManager.regions( fieldName ) );

  // Create a "fake" fine matrix (no data, just correct sizes/comms for use at coarse level init)
  localIndex const numLocalDof = dofManager.numLocalDofs( fieldName );
  m_matrix.createWithLocalSize( numLocalDof, numLocalDof, 0, comm );
}

namespace
{

template< typename Matrix >
Matrix filterMatrix( Matrix const & fineMatrix,
                     integer const numComp )
{
  GEOSX_MARK_SCOPE( filter );

  // 1. Apply SC approximation
  Matrix filteredMatrix;
  fineMatrix.separateComponentFilter( filteredMatrix, numComp );

  // 2. Filter out positive off-diagonal elements (assumed positive diagonals)
  filteredMatrix.clampEntries( -LvArray::NumericLimits< real64 >::infinity, 0.0, true );

  // 3. Enforce rowsum = 0
  // 3.1. Compute rowsums
  typename Matrix::Vector rowSums;
  rowSums.create( fineMatrix.numLocalRows(), fineMatrix.comm() );
  filteredMatrix.getRowSums( rowSums, RowSumType::SumValues );

  // 3.2. Preserve Dirichlet rows by setting the diagonal update to zero
  typename Matrix::Vector diag;
  diag.create( fineMatrix.numLocalRows(), fineMatrix.comm() );
  filteredMatrix.extractDiagonal( diag );
  forAll< parallelHostPolicy >( diag.localSize(), [diagData = diag.values(),
                                                   rowSumData = rowSums.open()]( localIndex const localRow )
  {
    if( isEqual( diagData[localRow], rowSumData[localRow] ) )
    {
      rowSumData[localRow] = 0.0;
    }
  } );
  rowSums.close();

  // 3.3. Subtract the nonzero rowsums from diagonal elements
  filteredMatrix.addDiagonal( rowSums, -1.0 );

  return filteredMatrix;
}

template< typename MATRIX >
auto makeJacobiMatrix( MATRIX && fineMatrix,
                       real64 const omega )
{
  GEOSX_MARK_SCOPE( jacobi );
  using Matrix = std::remove_const_t< TYPEOFREF( fineMatrix ) >;

  // 0. Copy or move input matrix into a new object
  Matrix iterMatrix( std::forward< MATRIX >( fineMatrix ) );

  // 1. Compute -w * Dinv * A;
  typename Matrix::Vector diag;
  diag.create( iterMatrix.numLocalRows(), iterMatrix.comm() );
  iterMatrix.extractDiagonal( diag );
  diag.reciprocal();
  diag.scale( -omega );
  iterMatrix.leftScale( diag );

  // 2. Compute I - w * Dinv * A by adding identity diagonal
  diag.set( 1.0 );
  iterMatrix.addDiagonal( diag, 1.0 );
  return iterMatrix;
}

template< typename Matrix >
Matrix makeIterationMatrix( Matrix const & fineMatrix,
                            integer const numComp,
                            real64 const omega,
                            integer const debugLevel,
                            string const & debugPrefix )
{
  GEOSX_MARK_SCOPE( make_iter_matrix );

  Matrix filteredMatrix = filterMatrix( fineMatrix, numComp );
  if( debugLevel >= 4 )
  {
    filteredMatrix.write( debugPrefix + "_filtered.mtx", LAIOutputFormat::MATRIX_MARKET );
  }

  Matrix jacobiMatrix = makeJacobiMatrix( std::move( filteredMatrix ), omega );
  if( debugLevel >= 4 )
  {
    jacobiMatrix.write( debugPrefix + "_jacobi.mtx", LAIOutputFormat::MATRIX_MARKET );
  }

  return jacobiMatrix;
}

template< typename Matrix >
integer iterateBasis( Matrix const & jacobiMatrix,
                      arrayView1d< globalIndex const > const & boundaryDof,
                      arrayView1d< globalIndex const > const & interiorDof,
                      integer const maxIter,
                      real64 const tolerance,
                      integer const checkFreq,
                      integer const debugLevel,
                      string const & name,
                      Matrix & prolongation )
{
  GEOSX_MARK_SCOPE( iterate );

  auto const saveForDebug = [&]( string const & suffix, integer const minDebugLevel )
  {
    if( debugLevel >= minDebugLevel )
    {
      prolongation.write( GEOSX_FMT( "{}_P_{}.mtx", name, suffix ), LAIOutputFormat::MATRIX_MARKET );
    }
  };

  Matrix P( prolongation );
  integer iter = 0;
  real64 norm = LvArray::NumericLimits< real64 >::max;

  auto const computeAndLogConvergenceNorm = [&]()
  {
    GEOSX_MARK_SCOPE( check );
    P.addEntries( prolongation, MatrixPatternOp::Same, -1.0 );
    norm = P.normMax( interiorDof );
    GEOSX_LOG_RANK_0_IF( debugLevel >= 3, GEOSX_FMT( "[MsRSB] {}: iter = {}, conv = {:e}", name, iter, norm ) );
  };

  saveForDebug( "init", 4 );
  while( iter < maxIter && norm > tolerance )
  {
    // Keep 1-based iteration index for convenience
    ++iter;

    // Perform a step of Jacobi
    Matrix Ptemp;
    {
      GEOSX_MARK_SCOPE( multiply );
      jacobiMatrix.multiply( prolongation, Ptemp );
    }

    // Restrict to the predefined prolongation pattern
    {
      GEOSX_MARK_SCOPE( restrict );
      P.zero();
      P.addEntries( Ptemp, MatrixPatternOp::Restrict, 1.0 );
    }

    // Rescale to preserve partition of unity
    {
      GEOSX_MARK_SCOPE( rescale );
      P.rescaleRows( boundaryDof, RowSumType::SumValues );
    }

    // Switch over to new prolongation operator
    std::swap( P, prolongation );
    saveForDebug( std::to_string( iter ), 6 );

    // Compute update norm, check convergence
    if( iter % checkFreq == 0 )
    {
      computeAndLogConvergenceNorm();
    }
  }

  // Compute update norm and check convergence one final time if needed (in case we ran out of iterations)
  if( iter % checkFreq != 0 )
  {
    computeAndLogConvergenceNorm();
  }

  GEOSX_LOG_RANK_0_IF( debugLevel >= 1, GEOSX_FMT( "[MsRSB] {}: {} in {} iterations", name, norm <= tolerance ? "converged" : "failed to converge", iter ) );

  saveForDebug( "conv", 4 );
  return iter;
}

} // namespace

template< typename LAI >
void MsrsbLevelBuilder< LAI >::compute( Matrix const & fineMatrix )
{
  GEOSX_MARK_FUNCTION;

  // Compute prolongation
  GEOSX_LOG_RANK_0_IF( m_params.debugLevel >= 2, GEOSX_FMT( "[MsRSB] {}: building iteration matrix", m_name ) );
  Matrix const jacobiMatrix = makeIterationMatrix( fineMatrix,
                                                   m_numComp,
                                                   m_params.msrsb.relaxation,
                                                   m_params.debugLevel,
                                                   m_name );

  GEOSX_LOG_RANK_0_IF( m_params.debugLevel >= 2, GEOSX_FMT( "[MsRSB] {}: performing basis iteration", m_name ) );
  m_lastNumIter = iterateBasis( jacobiMatrix,
                                m_boundaryDof,
                                m_interiorDof,
                                m_params.msrsb.maxIter,
                                m_params.msrsb.tolerance,
                                m_lastNumIter <= m_params.msrsb.checkFrequency ? 1 : m_params.msrsb.checkFrequency,
                                m_params.debugLevel,
                                m_name,
                                m_prolongation );

  if( m_params.debugLevel >= 5 )
  {
    writeProlongationForDebug();
  }

  // Recompute coarse operator - only if prolongation took a nontrivial number of iterations to converge
  if( m_lastNumIter > 1 || !m_matrix.ready() )
  {
    GEOSX_MARK_SCOPE( RAP );
    GEOSX_LOG_RANK_0_IF( m_params.debugLevel >= 2, GEOSX_FMT( "[MsRSB] {}: computing RAP", m_name ) );
    if( m_params.galerkin )
    {
      fineMatrix.multiplyPtAP( m_prolongation, m_matrix );
    }
    else
    {
      Matrix const & restriction = dynamicCast< Matrix const & >( *m_restriction );
      fineMatrix.multiplyRAP( m_prolongation, restriction, m_matrix );
    }
  }

  if( m_params.debugLevel >= 4 )
  {
    fineMatrix.write( m_name + "_fine.mtx", LAIOutputFormat::MATRIX_MARKET );
    m_matrix.write( m_name + "_coarse.mtx", LAIOutputFormat::MATRIX_MARKET );
  }
}

template< typename LAI >
void MsrsbLevelBuilder< LAI >::writeProlongationForDebug() const
{
  if( m_location == DofManager::Location::Node )
  {
    msrsb::writeProlongation( m_prolongation,
                              m_name,
                              m_numComp,
                              *m_mesh.fineMesh()->domain(),
                              m_mesh.fineMesh()->nodeManager(),
                              [&]( std::vector< string > const & names )
                              { m_mesh.fineMesh()->writeNodeData( names ); } );
  }
  else
  {
    msrsb::writeProlongation( m_prolongation,
                              m_name,
                              m_numComp,
                              *m_mesh.fineMesh()->domain(),
                              m_mesh.fineMesh()->cellManager(),
                              [&]( std::vector< string > const & names )
                              { m_mesh.fineMesh()->writeCellData( names ); } );
  }
}

// -----------------------
// Explicit Instantiations
// -----------------------
#ifdef GEOSX_USE_TRILINOS
template class MsrsbLevelBuilder< TrilinosInterface >;
#endif

#ifdef GEOSX_USE_HYPRE
template class MsrsbLevelBuilder< HypreInterface >;
#endif

#ifdef GEOSX_USE_PETSC
template class MsrsbLevelBuilder< PetscInterface >;
#endif

} // namespace multiscale
} // namespace geosx
