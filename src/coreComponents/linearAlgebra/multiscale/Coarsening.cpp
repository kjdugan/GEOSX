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
 * @file Coarsening.cpp
 */

#include "Coarsening.hpp"

#include "linearAlgebra/multiscale/MeshData.hpp"
#include "linearAlgebra/multiscale/MeshLevel.hpp"
#include "linearAlgebra/multiscale/MeshUtils.hpp"
#include "linearAlgebra/multiscale/PartitionerBase.hpp"
#include "mesh/DomainPartition.hpp"
#include "mesh/mpiCommunications/CommunicationTools.hpp"

#define ALLOW_MULTI_NODES 1

namespace geosx
{
namespace multiscale
{
namespace coarsening
{

namespace
{

void buildFineCellLists( MeshObjectManager const & fineCellManager,
                         MeshObjectManager & coarseCellManager )
{
  ArrayOfArrays< localIndex > & fineCellLists = coarseCellManager.getExtrinsicData< meshData::FineCellLocalIndices >();
  arrayView1d< localIndex const > const coarseCellLocalIndex = fineCellManager.getExtrinsicData< meshData::CoarseCellLocalIndex >();

  // Calculate the size of each list
  array1d< localIndex > fineCellListSizes( coarseCellManager.size() );
  forAll< parallelHostPolicy >( fineCellManager.size(), [=, sizes = fineCellListSizes.toView()] ( localIndex const ic )
  {
    RAJA::atomicInc( parallelHostAtomic{}, &sizes[coarseCellLocalIndex[ic]] );
  } );

  // Allocate space for each list
  fineCellLists.resizeFromCapacities< parallelHostPolicy >( fineCellListSizes.size(), fineCellListSizes.data() );

  // Populate the lists
  forAll< parallelHostPolicy >( fineCellManager.size(), [=, lists = fineCellLists.toView()] ( localIndex const ic )
  {
    lists.emplaceBackAtomic< parallelHostAtomic >( coarseCellLocalIndex[ic], ic );
  } );

  // Sort the lists for potentially better performance when using to sub-index larger arrays
  forAll< parallelHostPolicy >( coarseCellManager.size(), [lists = fineCellLists.toView()]( localIndex const icc )
  {
    arraySlice1d< localIndex > const list = lists[icc];
    LvArray::sortedArrayManipulation::makeSorted( list.begin(), list.end() );
  } );
}

void fillBasicCellData( MeshObjectManager const & fineCellManager,
                        MeshObjectManager & coarseCellManager )
{
  arrayView1d< localIndex const > const coarseCellIndex = fineCellManager.getExtrinsicData< meshData::CoarseCellLocalIndex >().toViewConst();
  arrayView1d< integer const > const fineGhostRank = fineCellManager.ghostRank();
  arrayView1d< integer const > const fineIsExternal = fineCellManager.isExternal();
  arrayView1d< integer const > const fineDomainBoundary = fineCellManager.getDomainBoundaryIndicator();

  arrayView1d< integer > const coarseGhostRank = coarseCellManager.ghostRank();
  arrayView1d< integer > const coarseIsExternal = coarseCellManager.isExternal();
  arrayView1d< integer > const coarseDomainBoundary = coarseCellManager.getDomainBoundaryIndicator();

  forAll< parallelHostPolicy >( fineCellManager.size(), [=]( localIndex const icf )
  {
    localIndex const icc = coarseCellIndex[icf];
    RAJA::atomicMax( parallelHostAtomic{}, &coarseGhostRank[icc], fineGhostRank[icf] );
    RAJA::atomicMax( parallelHostAtomic{}, &coarseIsExternal[icc], fineIsExternal[icf] );
    RAJA::atomicMax( parallelHostAtomic{}, &coarseDomainBoundary[icc], fineDomainBoundary[icf] );
  } );
}

void buildCellLocalToGlobalMaps( std::set< globalIndex > const & ghostGlobalIndices,
                                 globalIndex const rankOffset,
                                 MeshObjectManager & coarseCellManager )
{
  arrayView1d< globalIndex > const coarseLocalToGlobal = coarseCellManager.localToGlobalMap();
  {
    localIndex icc = 0;
    for(; icc < coarseCellManager.numOwnedObjects(); ++icc )
    {
      coarseLocalToGlobal[icc] = rankOffset + icc;
    }
    for( globalIndex const coarseGlobalIndex : ghostGlobalIndices )
    {
      coarseLocalToGlobal[icc++] = coarseGlobalIndex;
    }
  }
  coarseCellManager.constructGlobalToLocalMap();
  coarseCellManager.setMaxGlobalIndex();
}

template< typename T >
struct SetCompare
{
  ArrayOfSetsView< T const > const & sets;
  bool operator()( localIndex const i, localIndex const j ) const
  {
    arraySlice1d< T const > const si = sets[i];
    arraySlice1d< T const > const sj = sets[j];
    return std::lexicographical_compare( si.begin(), si.end(), sj.begin(), sj.end() );
  }
};

array1d< localIndex > findCoarseNodes( MeshObjectManager const & fineNodeManager,
                                       MeshObjectManager const & fineCellManager,
                                       ArrayOfSetsView< globalIndex const > const & nodeToSubdomain )
{
  // Construct a list of "skeleton" nodes (those with 3 or more adjacent subdomains)
  array1d< localIndex > skelNodes;
  for( localIndex inf = 0; inf < fineNodeManager.size(); ++inf )
  {
    if( nodeToSubdomain.sizeOfSet( inf ) >= 3 )
    {
      skelNodes.emplace_back( inf );
    }
  }

  // Sort skeleton nodes according to subdomain lists so as to locate nodes of identical adjacencies
  SetCompare< globalIndex > const adjacencyComp{ nodeToSubdomain.toViewConst() };
  std::sort( skelNodes.begin(), skelNodes.end(), adjacencyComp );

  // Identify "features" (groups of skeleton nodes with the same subdomain adjacency)
  array1d< localIndex > const featureIndex( fineNodeManager.size() );
  featureIndex.setValues< serialPolicy >( -1 );

  ArrayOfArrays< localIndex > featureNodes;
  featureNodes.reserve( skelNodes.size() ); // overallocate to avoid reallocation
  featureNodes.reserveValues( skelNodes.size() ); // precise allocation

  localIndex numFeatures = 0;
  featureNodes.appendArray( 0 );
  featureNodes.emplaceBack( numFeatures, skelNodes[0] );
  for( localIndex i = 1; i < skelNodes.size(); ++i )
  {
    if( adjacencyComp( skelNodes[i-1], skelNodes[i] ) )
    {
      ++numFeatures;
      featureNodes.appendArray( 0 );
    }
    featureNodes.emplaceBack( numFeatures, skelNodes[i] );
    featureIndex[skelNodes[i]] = numFeatures;
  }
  ++numFeatures;

  // Construct feature-to-feature adjacency
  ArrayOfSets< localIndex > const featureAdjacency( numFeatures, 64 );
  MeshObjectManager::MapViewConst const nodeToCell = fineNodeManager.toDualRelation().toViewConst();
  MeshObjectManager::MapViewConst const cellToNode = fineCellManager.toDualRelation().toViewConst();

  forAll< parallelHostPolicy >( numFeatures, [nodeToCell, cellToNode,
                                              featureNodes = featureNodes.toViewConst(),
                                              featureIndex = featureIndex.toViewConst(),
                                              featureAdjacency = featureAdjacency.toView()]( localIndex const f )
  {
    for( localIndex const inf : featureNodes[f] )
    {
      meshUtils::forUniqueNeighbors< 256 >( inf, nodeToCell, cellToNode, [&]( localIndex const nbrIdx )
      {
        if( nbrIdx != inf && featureIndex[nbrIdx] >= 0 )
        {
          featureAdjacency.insertIntoSet( f, featureIndex[nbrIdx] );
        }
      } );
    }
  } );

  // Choose features that represent coarse nodes (highest adjacency among neighbors)
  array1d< integer > const isCoarseNode( numFeatures );
  forAll< parallelHostPolicy >( numFeatures, [isCoarseNode = isCoarseNode.toView(),
                                              featureNodes = featureNodes.toViewConst(),
                                              featureAdjacency = featureAdjacency.toViewConst(),
                                              nodeToSubdomain = nodeToSubdomain.toViewConst()]( localIndex const f )
  {
    arraySlice1d< globalIndex const > const subs = nodeToSubdomain[ featureNodes( f, 0 ) ];
    for( localIndex const f_nbr : featureAdjacency[f] )
    {
      if( f_nbr != f )
      {
        arraySlice1d< globalIndex const > const subs_nbr = nodeToSubdomain[featureNodes( f_nbr, 0 )];
        if( std::includes( subs_nbr.begin(), subs_nbr.end(), subs.begin(), subs.end() ) )
        {
          // discard feature if its subdomain adjacency is fully included in any of its direct neighbors
          return;
        }
      }
    }
    // if not discarded, it is a coarse node
    isCoarseNode[f] = 1;
  } );

  // Make a list of fine-scale indices of coarse nodes that are locally owned
  array1d< localIndex > coarseNodes;
  for( localIndex f = 0; f < numFeatures; ++f )
  {
    if( isCoarseNode[f] == 1 )
    {
      arraySlice1d< localIndex const > const nodes = featureNodes[f];
#if ALLOW_MULTI_NODES
      for( localIndex inf : nodes )
      {
        coarseNodes.emplace_back( inf );
      }
#else
      coarseNodes.emplace_back( nodes[0] );
#endif
    }
  }

  std::sort( coarseNodes.begin(), coarseNodes.end() );
  return coarseNodes;
}

void fillBasicNodeData( MeshObjectManager const & fineNodeManager,
                        MeshObjectManager & coarseNodeManager )
{
  arrayView1d< localIndex const > const fineNodeIndex = coarseNodeManager.getExtrinsicData< meshData::FineNodeLocalIndex >().toViewConst();

  meshUtils::fillArrayBySrcIndex< parallelHostPolicy >( fineNodeIndex,
                                                        coarseNodeManager.ghostRank(),
                                                        fineNodeManager.ghostRank() );
  meshUtils::fillArrayBySrcIndex< parallelHostPolicy >( fineNodeIndex,
                                                        coarseNodeManager.isExternal(),
                                                        fineNodeManager.isExternal() );
  meshUtils::fillArrayBySrcIndex< parallelHostPolicy >( fineNodeIndex,
                                                        coarseNodeManager.getDomainBoundaryIndicator(),
                                                        fineNodeManager.getDomainBoundaryIndicator() );
}

ArrayOfSets< globalIndex >
buildFineNodeToGlobalSubdomainMap( MeshObjectManager const & fineNodeManager,
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
  ArrayOfSets< globalIndex > nodeToSubdomain;
  nodeToSubdomain.resizeFromCapacities< parallelHostPolicy >( rowCounts.size(), rowCounts.data() );

  // Fill the map
  arrayView1d< globalIndex const > const coarseCellGlobalIndex = fineCellManager.getExtrinsicData< meshData::CoarseCellGlobalIndex >();
  forAll< parallelHostPolicy >( fineNodeManager.size(), [=, nodeToSub = nodeToSubdomain.toView()]( localIndex const inf )
  {
    for( localIndex const icf : nodeToCell[inf] )
    {
      nodeToSub.insertIntoSet( inf, coarseCellGlobalIndex[icf] );
    }
  } );
  globalIndex numSubdomains = coarseCellManager.maxGlobalIndex() + 1;
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

void buildNodeToCellMap( MeshObjectManager const & fineCellManager,
                         MeshObjectManager const & fineNodeManager,
                         MeshObjectManager & coarseNodeManager )
{
  GEOSX_MARK_FUNCTION;

  arrayView1d< localIndex const > const coarseCellLocalIndex = fineCellManager.getExtrinsicData< meshData::CoarseCellLocalIndex >();
  arrayView1d< localIndex const > const fineNodeLocalIndex = coarseNodeManager.getExtrinsicData< meshData::FineNodeLocalIndex >();
  MeshObjectManager::MapViewConst const fineNodeToCell = fineNodeManager.toDualRelation().toViewConst();

  // First pass: count length of each sub-array in order to do exact allocation
  array1d< localIndex > rowCounts( coarseNodeManager.size() );
  forAll< parallelHostPolicy >( coarseNodeManager.size(), [=, rowCounts = rowCounts.toView()]( localIndex const inc )
  {
    localIndex count = 0;
    meshUtils::forUniqueNeighborValues< 256 >( fineNodeLocalIndex[inc],
                                               fineNodeToCell,
                                               coarseCellLocalIndex,
                                               []( auto ){ return true; },
                                               [&]( localIndex const )
    {
      ++count;
    } );
    rowCounts[inc] = count;
  } );

  // Resize the map
  MeshObjectManager::MapType & coarseNodeToCell = coarseNodeManager.toDualRelation();
  coarseNodeToCell.resizeFromCapacities< parallelHostPolicy >( rowCounts.size(), rowCounts.data() );

  // Second pass: fill the map
  forAll< parallelHostPolicy >( coarseNodeManager.size(), [=, coarseNodeToCell = coarseNodeToCell.toView()]( localIndex const inc )
  {
    meshUtils::forUniqueNeighborValues< 256 >( fineNodeLocalIndex[inc],
                                               fineNodeToCell,
                                               coarseCellLocalIndex,
                                               []( auto ){ return true; },
                                               [&]( localIndex const icc )
    {
      coarseNodeToCell.insertIntoSet( inc, icc );
    } );
  } );
}

void buildCellToNodeMap( MeshObjectManager const & coarseNodeManager,
                         MeshObjectManager & coarseCellManager )
{
  GEOSX_MARK_FUNCTION;

  MeshObjectManager::MapViewConst const nodeToCell = coarseNodeManager.toDualRelation().toViewConst();
  MeshObjectManager::MapType & cellToNode = coarseCellManager.toDualRelation();

  // First pass: count the row lengths in transpose map
  array1d< localIndex > rowCounts( cellToNode.size() );
  forAll< parallelHostPolicy >( nodeToCell.size(), [=, rowCounts = rowCounts.toView()]( localIndex const inc )
  {
    for( localIndex const icc : nodeToCell[inc] )
    {
      RAJA::atomicInc< parallelHostAtomic >( &rowCounts[icc] );
    }
  } );

  // Create and resize the temporary map
  ArrayOfArrays< localIndex > cellToNodeTemp;
  cellToNodeTemp.resizeFromCapacities< parallelHostPolicy >( rowCounts.size(), rowCounts.data() );

  // Second pass: fill the map
  forAll< parallelHostPolicy >( nodeToCell.size(), [=, cellToNode = cellToNodeTemp.toView()]( localIndex const inc )
  {
    for( localIndex const icc : nodeToCell[inc] )
    {
      cellToNode.emplaceBackAtomic< parallelHostAtomic >( icc, inc );
    }
  } );

  // Move the temp map and sort the entries
  cellToNode.assimilate( std::move( cellToNodeTemp ), LvArray::sortedArrayManipulation::UNSORTED_NO_DUPLICATES );
}

void buildCoarseCells( multiscale::MeshLevel & fineMesh,
                       multiscale::MeshLevel & coarseMesh,
                       LinearSolverParameters::Multiscale::Coarsening const & coarse_params )
{
  GEOSX_MARK_FUNCTION;

  MeshObjectManager & fineCellManager = fineMesh.cellManager();
  MeshObjectManager & coarseCellManager = coarseMesh.cellManager();

  // Allocate arrays to hold partitioning info (local and global)
  arrayView1d< localIndex > const & coarseLocalIndex =
    fineCellManager.registerExtrinsicData< meshData::CoarseCellLocalIndex >( coarseMesh.name() ).referenceAsView();
  arrayView1d< globalIndex > const & coarseGlobalIndex =
    fineCellManager.registerExtrinsicData< meshData::CoarseCellGlobalIndex >( coarseMesh.name() ).referenceAsView();

  // Generate the partitioning locally
  std::unique_ptr< PartitionerBase > partitioner = PartitionerBase::create( coarse_params );
  localIndex const numLocalCoarseCells = partitioner->generate( fineMesh, coarseLocalIndex );

  // Compute global number of partitions
  globalIndex const rankOffset = MpiWrapper::prefixSum< globalIndex >( numLocalCoarseCells );

  // Fill in partition global index for locally owned cells
  for( localIndex icf = 0; icf < fineCellManager.numOwnedObjects(); ++icf )
  {
    coarseGlobalIndex[icf] = coarseLocalIndex[icf] + rankOffset;
  }

  // Synchronize partition global index across ranks
  string_array fieldNames;
  fieldNames.emplace_back( meshData::CoarseCellGlobalIndex::key() );
  CommunicationTools::getInstance().synchronizeFields( fieldNames, fineCellManager, fineMesh.domain()->getNeighbors(), false );

  // Scan ghosted cells and collect all new partition global indices
  std::set< globalIndex > ghostGlobalIndices;
  for( localIndex icf = fineCellManager.numOwnedObjects(); icf < fineCellManager.size(); ++icf )
  {
    ghostGlobalIndices.insert( coarseGlobalIndex[icf] );
  }
  localIndex const numPresentCoarseCells = numLocalCoarseCells + LvArray::integerConversion< localIndex >( ghostGlobalIndices.size() );

  // Resize and start populating coarse cell manager
  coarseCellManager.resize( numPresentCoarseCells );
  coarseCellManager.setNumOwnedObjects( numLocalCoarseCells );

  // Compute cartesian indices for the coarse cells
  partitioner->setCoarseData( coarseMesh );

  // Populate coarse local-global maps
  buildCellLocalToGlobalMaps( ghostGlobalIndices, rankOffset, coarseCellManager );

  // finish filling the partition array for ghosted fine cells
  for( localIndex ic = fineCellManager.numOwnedObjects(); ic < fineCellManager.size(); ++ic )
  {
    coarseLocalIndex[ic] = coarseCellManager.globalToLocalMap( coarseGlobalIndex[ic] );
  }

  coarseCellManager.registerWrapper< meshData::FineCellLocalIndices::type >( meshData::FineCellLocalIndices::key() );
  buildFineCellLists( fineCellManager, coarseCellManager );
  fillBasicCellData( fineCellManager, coarseCellManager );

  // Populate neighbor data and sets
  std::vector< int > neighborRanks = fineMesh.domain()->getNeighborRanks();
  for( int const rank : neighborRanks )
  {
    coarseCellManager.addNeighbor( rank );
  }
  meshUtils::copySets( fineCellManager,
                       meshData::CoarseCellLocalIndex::key(),
                       coarseCellManager );
  meshUtils::copyNeighborData( fineCellManager,
                               meshData::CoarseCellLocalIndex::key(),
                               neighborRanks,
                               coarseCellManager,
                               meshUtils::filterArrayUnique< localIndex > );
}

void buildCoarseNodes( multiscale::MeshLevel & fineMesh,
                       multiscale::MeshLevel & coarseMesh,
                       arrayView1d< string const > const & boundaryNodeSets )
{
  GEOSX_MARK_FUNCTION;

  MeshObjectManager & fineNodeManager = fineMesh.nodeManager();
  MeshObjectManager & fineCellManager = fineMesh.cellManager();
  MeshObjectManager & coarseNodeManager = coarseMesh.nodeManager();
  MeshObjectManager & coarseCellManager = coarseMesh.cellManager();

  // Create the array manually in order to specify default capacity (no suitable post-resize facility exists)
  ArrayOfSets< globalIndex > & nodeToSubdomain =
    fineNodeManager.registerWrapper< meshData::NodeToCoarseSubdomain::type >( meshData::NodeToCoarseSubdomain::key() ).reference();

  // Build and sync an adjacency map of fine nodes to coarse subdomains (including global boundaries)
  nodeToSubdomain = buildFineNodeToGlobalSubdomainMap( fineNodeManager, fineCellManager, coarseCellManager, boundaryNodeSets );
  array1d< string > fields;
  fields.emplace_back( meshData::NodeToCoarseSubdomain::key() );
  CommunicationTools::getInstance().synchronizeFields( fields, fineNodeManager, fineMesh.domain()->getNeighbors(), false );

  // Find all locally present coarse nodes
  array1d< localIndex > const coarseNodes = findCoarseNodes( fineNodeManager, fineCellManager, nodeToSubdomain.toViewConst() );

  // Reorder them to have all local nodes precede ghosted (stable partition to preserve order)
  arrayView1d< integer const > const nodeGhostRank = fineNodeManager.ghostRank();
  auto const localEnd = std::stable_partition( coarseNodes.begin(), coarseNodes.end(),
                                               [=]( localIndex const inf ){ return nodeGhostRank[inf] < 0; } );

  localIndex const numLocalCoarseNodes = std::distance( coarseNodes.begin(), localEnd );
  localIndex const numPresentCoarseNodes = coarseNodes.size();
  globalIndex const rankOffset = MpiWrapper::prefixSum< globalIndex >( numLocalCoarseNodes );

  coarseNodeManager.resize( numPresentCoarseNodes );
  coarseNodeManager.setNumOwnedObjects( numLocalCoarseNodes );

  // Finally, build coarse node maps
  arrayView1d< localIndex > const coarseNodeLocalIndex =
    fineNodeManager.registerExtrinsicData< meshData::CoarseNodeLocalIndex >( coarseMesh.name() ).reference().toView();
  arrayView1d< globalIndex > const coarseNodeGlobalIndex =
    fineNodeManager.registerExtrinsicData< meshData::CoarseNodeGlobalIndex >( coarseMesh.name() ).reference().toView();
  arrayView1d< localIndex > const fineNodeLocalIndex =
    coarseNodeManager.registerExtrinsicData< meshData::FineNodeLocalIndex >( coarseMesh.name() ).reference().toView();
  arrayView1d< globalIndex > const coarseNodeLocalToGlobal =
    coarseNodeManager.localToGlobalMap();

  // Fill the local part
  forAll< parallelHostPolicy >( numLocalCoarseNodes, [=, coarseNodes = coarseNodes.toViewConst()]( localIndex const i )
  {
    localIndex const inf = coarseNodes[i];
    coarseNodeLocalIndex[inf] = i;
    coarseNodeGlobalIndex[inf] = rankOffset + i;
    coarseNodeLocalToGlobal[i] = rankOffset + i;
    fineNodeLocalIndex[i] = inf;
  } );

  // Sync across ranks
  string_array fieldNames;
  fieldNames.emplace_back( meshData::CoarseNodeGlobalIndex::key() );
  CommunicationTools::getInstance().synchronizeFields( fieldNames, fineNodeManager, fineMesh.domain()->getNeighbors(), false );

  // Fill the ghosted part
  forAll< parallelHostPolicy >( numLocalCoarseNodes, numPresentCoarseNodes,
                                [=, coarseNodes = coarseNodes.toViewConst()]( localIndex const i )
  {
    localIndex const inf = coarseNodes[i];
    coarseNodeLocalIndex[inf] = i;
    coarseNodeLocalToGlobal[i] = coarseNodeGlobalIndex[inf];
    fineNodeLocalIndex[i] = inf;
  } );

  coarseNodeManager.constructGlobalToLocalMap();
  coarseNodeManager.setMaxGlobalIndex();
  fillBasicNodeData( fineNodeManager, coarseNodeManager );

  // Populate neighbor data and sets
  std::vector< int > neighborRanks = fineMesh.domain()->getNeighborRanks();
  for( int const rank : neighborRanks )
  {
    coarseNodeManager.addNeighbor( rank );
  }
  meshUtils::copySets( fineNodeManager,
                       meshData::CoarseNodeLocalIndex::key(),
                       coarseNodeManager );
  meshUtils::copyNeighborData( fineNodeManager,
                               meshData::CoarseNodeLocalIndex::key(),
                               neighborRanks,
                               coarseNodeManager,
                               meshUtils::filterArray< localIndex > );
}

} // namespace

void buildCoarseMesh( MeshLevel & fineMesh,
                      MeshLevel & coarseMesh,
                      LinearSolverParameters::Multiscale::Coarsening const & coarse_params,
                      array1d< string > const & boundaryNodeSets )
{
  buildCoarseCells( fineMesh, coarseMesh, coarse_params );
  buildCoarseNodes( fineMesh, coarseMesh, boundaryNodeSets );
  buildNodeToCellMap( fineMesh.cellManager(), fineMesh.nodeManager(), coarseMesh.nodeManager() );
  buildCellToNodeMap( coarseMesh.nodeManager(), coarseMesh.cellManager() );
}

} // namespace coarsening
} // namespace multiscale
} // namespace geosx
