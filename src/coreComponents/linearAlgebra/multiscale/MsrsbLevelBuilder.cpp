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
#include "linearAlgebra/utilities/LAIHelperFunctions.hpp"
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

namespace
{

template< typename LAI >
std::unique_ptr< LinearOperator< typename LAI::ParallelVector > >
makeRestriction( LinearSolverParameters::Multiscale const & params,
                 typename LAI::ParallelMatrix const & prolongation )
{
  std::unique_ptr< LinearOperator< typename LAI::ParallelVector > > restriction;
  if( params.galerkin )
  {
    // Make a transpose operator with a reference to P, which will be computed later
    restriction = std::make_unique< TransposeOperator< LAI > >( prolongation );
  }
  else
  {
    // Make an explicit transpose of tentative (initial) P
    typename LAI::ParallelMatrix R;
    prolongation.transpose( R );
    restriction = std::make_unique< typename LAI::ParallelMatrix >( std::move( R ) );
  }
  return restriction;
}

ArrayOfSets< localIndex >
buildFineNodeToLocalSubdomainMap( MeshObjectManager const & fineNodeManager,
                                  MeshObjectManager const & fineCellManager,
                                  MeshObjectManager const & coarseCellManager,
                                  arrayView1d< string const > const & boundaryNodeSets )
{
  MeshObjectManager::MapViewConst const nodeToCell = fineNodeManager.toDualRelation().toViewConst();

  // count the row lengths
  array1d< localIndex > rowCounts( fineNodeManager.size() );
  forAll< parallelHostPolicy >( fineNodeManager.size(), [=, rowCounts = rowCounts.toView()]( localIndex const inf )
  {
    rowCounts[inf] = nodeToCell.sizeOfSet( inf );
  } );
  for( string const & setName : boundaryNodeSets )
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
  arrayView1d< localIndex const > const coarseCellLocalIndex = fineCellManager.getExtrinsicData< meshData::CoarseCellLocalIndex >();
  forAll< parallelHostPolicy >( fineNodeManager.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inf )
  {
    for( localIndex const icf : nodeToCell[inf] )
    {
      nodeToSub.insertIntoSet( inf, coarseCellLocalIndex[icf] );
    }
  } );
  localIndex numSubdomains = coarseCellManager.size();
  for( string const & setName : boundaryNodeSets )
  {
    SortedArrayView< localIndex const > const set = fineNodeManager.getSet( setName ).toViewConst();
    forAll< parallelHostPolicy >( set.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inf )
    {
      nodeToSub.insertIntoSet( set[inf], numSubdomains );
    } );
    ++numSubdomains;
  }

  return nodeToSubdomain;
}

ArrayOfSets< localIndex >
buildCoarseNodeToLocalSubdomainMap( MeshObjectManager const & coarseNodeManager,
                                    MeshObjectManager const & coarseCellManager,
                                    arrayView1d< string const > const & boundaryNodeSets )
{
  MeshObjectManager::MapViewConst const nodeToCell = coarseNodeManager.toDualRelation().toViewConst();

  // count the row lengths
  array1d< localIndex > rowCounts( coarseNodeManager.size() );
  forAll< parallelHostPolicy >( coarseNodeManager.size(), [=, rowCounts = rowCounts.toView()]( localIndex const inc )
  {
    rowCounts[inc] = nodeToCell.sizeOfSet( inc );
  } );
  for( string const & setName : boundaryNodeSets )
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
  forAll< parallelHostPolicy >( coarseNodeManager.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inc )
  {
    nodeToSub.insertIntoSet( inc, nodeToCell[inc].begin(), nodeToCell[inc].end() );
  } );
  localIndex numSubdomains = coarseCellManager.size();
  for( string const & setName : boundaryNodeSets )
  {
    SortedArrayView< localIndex const > const set = coarseNodeManager.getSet( setName ).toViewConst();
    forAll< parallelHostPolicy >( set.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inc )
    {
      nodeToSub.insertIntoSet( set[inc], numSubdomains );
    } );
    ++numSubdomains;
  }

  return nodeToSubdomain;
}

/**
 * @brief Build the basic sparsity pattern for prolongation.
 *
 * Support of a coarse nodal basis function is defined as the set of fine-scale nodes
 * that are adjacent exclusively to subdomains (coarse cells or boundaries) that are
 * also adjacent to that coarse node.
 */
void
buildNodalSupports( multiscale::MeshLevel const & fine,
                    multiscale::MeshLevel const & coarse,
                    arrayView1d< string const > const & boundaryNodeSets,
                    ArrayOfSets< localIndex > & supports,
                    arrayView1d< integer > const & supportBoundaryIndicator )
{
  GEOSX_MARK_FUNCTION;

  ArrayOfSets< localIndex > const fineNodeToCoarseCell =
    buildFineNodeToLocalSubdomainMap( fine.nodeManager(), fine.cellManager(), coarse.cellManager(), {} );
  ArrayOfSets< localIndex > const fineNodeToSubdomain =
    buildFineNodeToLocalSubdomainMap( fine.nodeManager(), fine.cellManager(), coarse.cellManager(), boundaryNodeSets );
  ArrayOfSets< localIndex > const coarseNodeToSubdomain =
    buildCoarseNodeToLocalSubdomainMap( coarse.nodeManager(), coarse.cellManager(), boundaryNodeSets );

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

  // Count row lengths
  array1d< localIndex > rowLengths( fine.nodeManager().size() );
  forAll< parallelHostPolicy >( fine.nodeManager().size(), [coarseNodeIndex, coarseCellToNode,
                                                            rowLengths = rowLengths.toView(),
                                                            fineNodeToSubdomain = fineNodeToSubdomain.toViewConst(),
                                                            fineNodeToCoarseCell = fineNodeToCoarseCell.toViewConst(),
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
      meshUtils::forUniqueNeighbors< 256 >( inf, fineNodeToCoarseCell, coarseCellToNode, [&]( localIndex )
      {
        ++numCoarseNodes;
      } );
      rowLengths[inf] = numCoarseNodes;
      supportBoundaryIndicator[inf] = 1;
    }
  } );

  // Resize
  supports.resizeFromCapacities< parallelHostPolicy >( rowLengths.size(), rowLengths.data() );

  // Fill the map
  forAll< parallelHostPolicy >( fine.nodeManager().size(), [coarseNodeIndex, coarseCellToNode,
                                                            supports = supports.toView(),
                                                            fineNodeToSubdomain = fineNodeToSubdomain.toViewConst(),
                                                            fineNodeToCoarseCell = fineNodeToCoarseCell.toViewConst(),
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
      meshUtils::forUniqueNeighbors< 256 >( inf, fineNodeToCoarseCell, coarseCellToNode, [&]( localIndex const inc )
      {
        arraySlice1d< localIndex const > const csubs = coarseNodeToSubdomain[inc];
        if( std::includes( csubs.begin(), csubs.end(), fsubs.begin(), fsubs.end() ) )
        {
          supports.insertIntoSet( inf, inc );
        }
      } );
    }
  } );
}

SparsityPattern< globalIndex >
buildProlongationSparsity( MeshObjectManager const & fineManager,
                           MeshObjectManager const & coarseManager,
                           ArrayOfSetsView< localIndex const > const & supports,
                           integer const numComp )
{
  GEOSX_MARK_FUNCTION;

  // This assumes an SDC-type pattern (i.e. no coupling between dof components on the same node)
  array1d< localIndex > rowLengths( numComp * fineManager.numOwnedObjects() );
  forAll< parallelHostPolicy >( fineManager.numOwnedObjects(), [=, rowLengths = rowLengths.toView()]( localIndex const k )
  {
    for( integer ic = 0; ic < numComp; ++ic )
    {
      rowLengths[k * numComp + ic] = supports.sizeOfSet( k );
    }
  } );

  SparsityPattern< globalIndex > pattern;
  pattern.resizeFromRowCapacities< parallelHostPolicy >( rowLengths.size(),
                                                         numComp * ( coarseManager.maxGlobalIndex() + 1 ),
                                                         rowLengths.data() );

  arrayView1d< globalIndex const > const coarseLocalToGlobal = coarseManager.localToGlobalMap();
  forAll< parallelHostPolicy >( fineManager.numOwnedObjects(), [=, pattern = pattern.toView()]( localIndex const k )
  {
    localIndex const fineOffset = k * numComp;
    for( localIndex const inc : supports[k] )
    {
      globalIndex const coarseOffset = coarseLocalToGlobal[inc] * numComp;
      for( integer ic = 0; ic < numComp; ++ic )
      {
        pattern.insertNonZero( fineOffset + ic, coarseOffset + ic );
      }
    }
  } );

  return pattern;
}

ArrayOfSets< localIndex >
buildLocalNodalConnectivity( MeshObjectManager const & nodeManager,
                             MeshObjectManager const & cellManager )
{
  MeshObjectManager::MapViewConst const nodeToCell = nodeManager.toDualRelation().toViewConst();
  MeshObjectManager::MapViewConst const cellToNode = cellManager.toDualRelation().toViewConst();
  arrayView1d< integer const > const cellGhostRank = cellManager.ghostRank();

  // Collect row sizes
  array1d< localIndex > rowLength( nodeManager.size() );
  forAll< parallelHostPolicy >( nodeManager.size(), [=, rowLength = rowLength.toView()]( localIndex const k )
  {
    meshUtils::forUniqueNeighbors< 256 >( k, nodeToCell, cellToNode, cellGhostRank, [&]( localIndex const )
    {
      ++rowLength[k];
    } );
  } );

  // Resize
  ArrayOfSets< localIndex > conn( nodeManager.size() );
  conn.resizeFromCapacities< parallelHostPolicy >( rowLength.size(), rowLength.data() );

  // Fill the map
  forAll< parallelHostPolicy >( nodeManager.size(), [=, conn = conn.toView()]( localIndex const k )
  {
    meshUtils::forUniqueNeighbors< 256 >( k, nodeToCell, cellToNode, cellGhostRank, [&]( localIndex const n )
    {
      conn.insertIntoSet( k, n );
    } );
  } );

  return conn;
}

array1d< localIndex >
makeSeededPartition( ArrayOfSetsView< localIndex const > const & connectivity,
                     arrayView1d< localIndex const > const & seeds,
                     ArrayOfSetsView< localIndex const > const & supports )
{
  localIndex const numParts = seeds.size();
  localIndex const numNodes = connectivity.size();

  array1d< localIndex > part( numNodes );
  part.setValues< parallelHostPolicy >( -1 );

  // Initialize the partitions and expansion front
  array1d< localIndex > front;
  front.reserve( numNodes );
  forAll< serialPolicy >( numParts, [&]( localIndex const ip )
  {
    part[seeds[ip]] = ip;
    for( localIndex const k : connectivity[seeds[ip]] )
    {
      if( part[k] < 0 && supports.contains( k, ip ) )
      {
        front.emplace_back( k );
      }
    }
  } );

  // Use AoA with 1 array for its atomic emplace capability
  // TODO: this might not be efficient due to thread contention;
  //       may need to use a serial policy instead (benchmark me)
  ArrayOfArrays< localIndex > unassigned( 1, 12 * numNodes );
  array1d< localIndex > newPart;

  integer numIter = 0;
  while( !front.empty() )
  {
    // Make the list unique
    localIndex const numFrontNodes = LvArray::sortedArrayManipulation::makeSortedUnique( front.begin(), front.end() );
    front.resize( numFrontNodes );
    newPart.resize( numFrontNodes );
    newPart.setValues< parallelHostPolicy >( -1 );

    // Pass 1: assign partitions to the front nodes based on majority among neighbors
    RAJA::ReduceSum< parallelHostReduce, localIndex > numAssn = 0;
    forAll< parallelHostPolicy >( front.size(), [connectivity, supports, numAssn,
                                                 front = front.toViewConst(),
                                                 part = part.toViewConst(),
                                                 newPart = newPart.toView()]( localIndex const i )
    {
      localIndex const k = front[i];
      localIndex maxCount = 0;
      meshUtils::forUniqueNeighborValues< 256 >( k, connectivity, part,
                                                 []( localIndex const p ){ return p >= 0; }, // only assigned nodes
                                                 [&]( localIndex const p, localIndex const count )
      {
        if( p >= 0 && count > maxCount && supports.contains( k, p ) )
        {
          newPart[i] = p;
          maxCount = count;
        }
      } );

      if( maxCount > 0 )
      {
        numAssn += 1;
      }
    } );

    // Terminate the loop as soon as no new assignments are made
    if( numAssn.get() == 0 )
    {
      break;
    }

    // Pass 2: copy new front partition assignments back into full partition array
    forAll< parallelHostPolicy >( front.size(), [front = front.toViewConst(),
                                                 newPart = newPart.toViewConst(),
                                                 part = part.toView()] ( localIndex const i )
    {
      part[front[i]] = newPart[i];
    } );

    // Pass 3: build a list of unassigned neighbor indices to become the new front
    forAll< parallelHostPolicy >( front.size(), [connectivity,
                                                 front = front.toViewConst(),
                                                 part = part.toViewConst(),
                                                 unassigned = unassigned.toView()]( localIndex const i )
    {
      localIndex const k = front[i];
      meshUtils::forUniqueNeighborValues< 256 >( k, connectivity,
                                                 []( localIndex const _ ){ return _; }, // just unique neighbor indices
                                                 [&]( localIndex const n ){ return part[n] < 0; }, // only unassigned nodes
                                                 [&]( localIndex const n )
      {
        unassigned.emplaceBackAtomic< parallelHostAtomic >( 0, n );
      } );
    } );

    // Make the new front expansion list
    front.clear();
    front.insert( 0, unassigned[0].begin(), unassigned[0].end() );
    unassigned.clearArray( 0 );

    ++numIter;
  }

  GEOSX_WARNING_IF( !front.empty(), "[MsRSB]: nodes not assigned to initial partition: " << front );

  // Attempt to fix unassigned nodes, if any
  forAll< parallelHostPolicy >( front.size(), [=, front = front.toViewConst(),
                                               part = part.toView() ]
                                  ( localIndex const i )
  {
    localIndex const k = front[i];
    part[k] = supports( k, 0 ); // assign to the first support the node belongs to
  } );

  return part;
}

CRSMatrix< real64, globalIndex >
buildTentativeProlongation( multiscale::MeshLevel const & fineMesh,
                            multiscale::MeshLevel const & coarseMesh,
                            ArrayOfSetsView< localIndex const > const & supports,
                            integer const numComp )
{
  GEOSX_MARK_FUNCTION;

  // Build support regions and tentative prolongation
  ArrayOfSets< localIndex > const nodalConn = buildLocalNodalConnectivity( fineMesh.nodeManager(), fineMesh.cellManager() );
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
  forAll< parallelHostPolicy >( fineMesh.nodeManager().numOwnedObjects(), [=, localMatrix = localMatrix.toViewConstSizes()]( localIndex const inf )
  {
    if( initPart[inf] >= 0 )
    {
      real64 const value = 1.0;
      for( integer ic = 0; ic < numComp; ++ic )
      {
        globalIndex const col = coarseLocalToGlobal[initPart[inf]] * numComp + ic;
        localMatrix.addToRow< serialAtomic >( inf * numComp + ic, &col, &value, 1.0 );
      }
    }
  } );

  return localMatrix;
}

template< typename Matrix >
void plotProlongation( Matrix const & prolongation,
                       string const & prefix,
                       integer const numComp,
                       multiscale::MeshLevel & fineMesh )
{
  std::vector< string > bNames{ "X ", "Y ", "Z " };
  std::vector< string > cNames{ " x", " y", " z" };

  globalIndex const numNodes = prolongation.numGlobalCols() / numComp;
  int const paddedSize = LvArray::integerConversion< int >( std::to_string( numNodes ).size() );

  std::vector< arrayView3d< real64 > > views;
  std::vector< string > names;

  for( globalIndex icn = 0; icn < numNodes; ++icn )
  {
    string const name = prefix + "_P_" + stringutilities::padValue( icn, paddedSize );
    auto arr = std::make_unique< array3d< real64 > >( fineMesh.nodeManager().size(), numComp, numComp );
    auto & wrapper = fineMesh.nodeManager().registerWrapper( name, std::move( arr ) ).
                       setPlotLevel( dataRepository::PlotLevel::LEVEL_0 ).
                       setDimLabels( 1, { bNames.begin(), bNames.begin() + numComp } ).
                       setDimLabels( 2, { cNames.begin(), cNames.begin() + numComp } );
    views.push_back( wrapper.referenceAsView() );
    names.push_back( name );
  }

  array1d< globalIndex > colIndices;
  array1d< real64 > values;

  for( localIndex localRow = 0; localRow < prolongation.numLocalRows(); ++localRow )
  {
    globalIndex const globalRow = prolongation.ilower() + localRow;
    localIndex const rowNode = localRow / numComp;
    integer const rowComp = static_cast< integer >( localRow % numComp );

    localIndex const numValues = prolongation.rowLength( globalRow );
    colIndices.resize( numValues );
    values.resize( numValues );
    prolongation.getRowCopy( globalRow, colIndices, values );

    for( localIndex i = 0; i < numValues; ++i )
    {
      globalIndex const colNode = colIndices[i] / numComp;
      integer const colComp = static_cast< integer >( colIndices[i] % numComp );
      views[colNode]( rowNode, colComp, rowComp ) = values[i];
    }
  }

  string_array fieldNames;
  fieldNames.insert( 0, names.begin(), names.end() );
  CommunicationTools::getInstance().synchronizeFields( fieldNames, fineMesh.nodeManager(), fineMesh.domain()->getNeighbors(), false );

  fineMesh.writeNodeData( names );
  for( string const & name : names )
  {
    fineMesh.nodeManager().deregisterWrapper( name );
  }
}

void findSupportBoundaries( arrayView1d< integer const > const & indicator,
                            integer const numComp,
                            globalIndex const firstLocalDof,
                            array1d< globalIndex > & boundaryDof,
                            array1d< globalIndex > & interiorDof )
{
  boundaryDof.reserve( indicator.size() * numComp );
  interiorDof.reserve( indicator.size() * numComp );
  forAll< serialPolicy >( indicator.size(), [&]( localIndex const inf )
  {
    globalIndex const globalDof = firstLocalDof + inf * numComp;
    if( indicator[inf] )
    {
      for( integer c = 0; c < numComp; ++c )
      {
        boundaryDof.emplace_back( globalDof + c );
      }
    }
    else
    {
      for( integer c = 0; c < numComp; ++c )
      {
        interiorDof.emplace_back( globalDof + c );
      }
    }
  } );
}

} // namespace

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
  CRSMatrix< real64, globalIndex > localProlongation;
  {
    ArrayOfSets< localIndex > supports;
    array1d< integer > supportBoundaryIndicators( fineMgr.size() );
    buildNodalSupports( fine.mesh(), m_mesh, m_params.boundarySets, supports, supportBoundaryIndicators );
    supportBoundaryIndicators.resize( fineMgr.numOwnedObjects() ); // TODO fix proper
    localProlongation = buildTentativeProlongation( fine.mesh(), m_mesh, supports.toViewConst(), m_numComp );
    findSupportBoundaries( supportBoundaryIndicators, m_numComp, fine.matrix().ilower(), m_boundaryDof, m_interiorDof );
  }
  m_prolongation.create( localProlongation.toViewConst(), coarseMgr.numOwnedObjects() * m_numComp, fine.matrix().comm() );
  m_restriction = makeRestriction< LAI >( m_params, m_prolongation );
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
    plotProlongation( m_prolongation, m_name, m_numComp, *m_mesh.fineMesh() );
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
