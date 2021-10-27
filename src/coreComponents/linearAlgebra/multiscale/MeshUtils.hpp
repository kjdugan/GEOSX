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
 * @file MeshUtils.hpp
 */

#ifndef GEOSX_LINEARALGEBRA_MULTISCALE_MESHUTILS_HPP
#define GEOSX_LINEARALGEBRA_MULTISCALE_MESHUTILS_HPP

#include "common/DataTypes.hpp"
#include "mesh/ObjectManagerBase.hpp"

namespace geosx
{
namespace multiscale
{
namespace meshUtils
{

template< typename T, typename U = T >
void filterArray( arrayView1d< T const > const & src,
                  arrayView1d< U const > const & map,
                  array1d< U > & dst )
{
  dst.reserve( src.size() );
  for( T const & val : src )
  {
    U const newVal = map[val];
    if( newVal >= 0 )
    {
      dst.emplace_back( newVal );
    }
  }
}

template< typename T, typename U = T >
void filterArrayUnique( arrayView1d< T const > const & src,
                        arrayView1d< U const > const & map,
                        array1d< U > & dst )
{
  SortedArray< U > values;
  values.reserve( src.size() );
  for( T const & val : src )
  {
    U const newVal = map[val];
    if( newVal >= 0 )
    {
      values.insert( newVal );
    }
  }
  for( U const & val : values )
  {
    dst.emplace_back( val );
  }
}

template< typename T, typename U = T >
void filterSet( SortedArrayView< T const > const & src,
                arrayView1d< U const > const & map,
                SortedArray< U > & dst )
{
  dst.reserve( src.size() );
  for( T const & val : src )
  {
    U const newVal = map[val];
    if( newVal >= 0 )
    {
      dst.insert( newVal );
    }
  }
}

template< typename POLICY, typename T, typename INDEX, typename FUNC >
void fillArrayByDstIndex( arrayView1d< INDEX const > const & indices,
                          arrayView1d< T > const & dst,
                          FUNC src )
{
  forAll< POLICY >( indices.size(), [=]( INDEX const srcIdx )
  {
    INDEX const dstIdx = indices[srcIdx];
    if( dstIdx >= 0 )
    {
      dst[dstIdx] = src( srcIdx );
    }
  } );
}

template< typename POLICY, typename T, typename INDEX, typename FUNC >
void fillArrayBySrcIndex( arrayView1d< INDEX const > const & map,
                          arrayView1d< T > const & dst,
                          FUNC src )
{
  GEOSX_ASSERT_EQ( dst.size(), map.size() );
  forAll< POLICY >( map.size(), [=]( INDEX const i )
  {
    INDEX const k = map[i];
    if( k >= 0 )
    {
      dst[i] = src( k );
    }
  } );
}

template< typename FUNC >
void copyNeighborData( ObjectManagerBase const & srcManager,
                       string const & mapKey,
                       std::vector< integer > const & ranks,
                       ObjectManagerBase & dstManager,
                       FUNC copyFunc )
{
  arrayView1d< localIndex const > const map = srcManager.getReference< array1d< localIndex > >( mapKey );
  for( integer const rank : ranks )
  {
    NeighborData const & srcData = srcManager.getNeighborData( rank );
    NeighborData & dstData = dstManager.getNeighborData( rank );
    copyFunc( srcData.ghostsToSend(), map, dstData.ghostsToSend() );
    copyFunc( srcData.ghostsToReceive(), map, dstData.ghostsToReceive() );
    copyFunc( srcData.adjacencyList(), map, dstData.adjacencyList() );
    copyFunc( srcData.matchedPartitionBoundary(), map, dstData.matchedPartitionBoundary() );
  }
}

void copySets( ObjectManagerBase const & srcManager,
               string const & mapKey,
               ObjectManagerBase & dstManager );

namespace internal
{

IS_VALID_EXPRESSION_2( isCallableWithArg, T, U, std::declval< T >()( std::declval< U >() ) );
IS_VALID_EXPRESSION_2( isCallableWithArgAndIndex, T, U, std::declval< T >()( std::declval< U >(), localIndex() ) );

template< typename T, typename LAMBDA >
std::enable_if_t< isCallableWithArg< LAMBDA, T > >
forUniqueValuesHelper( T const & val, localIndex const count, LAMBDA lambda )
{
  GEOSX_UNUSED_VAR( count );
  lambda( val );
}

template< typename T, typename LAMBDA >
std::enable_if_t< isCallableWithArgAndIndex< LAMBDA, T > >
forUniqueValuesHelper( T const & val, localIndex const count, LAMBDA lambda )
{
  lambda( val, count );
}

} // namespace internal

template< typename T, typename LAMBDA >
void forUniqueValues( T * const ptr, localIndex const size, LAMBDA && lambda )
{
  if( size == 0 )
  {
    return;
  }
  LvArray::sortedArrayManipulation::makeSorted( ptr, ptr + size );

  localIndex numRepeatedValues = 0;
  for( localIndex i = 0; i < size - 1; ++i )
  {
    ++numRepeatedValues;
    if( ptr[i + 1] != ptr[i] )
    {
      internal::forUniqueValuesHelper( ptr[i], numRepeatedValues, std::forward< LAMBDA >( lambda ) );
      numRepeatedValues = 0;
    }
  }
  ++numRepeatedValues;
  internal::forUniqueValuesHelper( ptr[size - 1], numRepeatedValues, std::forward< LAMBDA >( lambda ) );
}

template< integer MAX_NEIGHBORS, typename L2C_MAP_TYPE, typename C2L_MAP_TYPE, typename LAMBDA >
void forUniqueNeighbors( localIndex const locIdx,
                         L2C_MAP_TYPE const & locToConn,
                         C2L_MAP_TYPE const & connToLoc,
                         LAMBDA && lambda )
{
  localIndex neighbors[MAX_NEIGHBORS];
  integer numNeighbors = 0;
  for( localIndex const connIdx : locToConn[locIdx] )
  {
    for( localIndex const nbrIdx : connToLoc[connIdx] )
    {
      GEOSX_ERROR_IF_LT( nbrIdx, 0 );
      GEOSX_ERROR_IF_GE_MSG( numNeighbors, MAX_NEIGHBORS, "Too many neighbors, need to increase stack limit" );
      neighbors[numNeighbors++] = nbrIdx;
    }
  }
  forUniqueValues( neighbors, numNeighbors, std::forward< LAMBDA >( lambda ) );
}

template< integer MAX_NEIGHBORS, typename L2C_MAP_TYPE, typename C2L_MAP_TYPE, typename LAMBDA >
void forUniqueNeighbors( localIndex const locIdx,
                         L2C_MAP_TYPE const & locToConn,
                         C2L_MAP_TYPE const & connToLoc,
                         arrayView1d< integer const > const & connGhostRank,
                         LAMBDA && lambda )
{
  localIndex neighbors[MAX_NEIGHBORS];
  integer numNeighbors = 0;
  for( localIndex const connIdx : locToConn[locIdx] )
  {
    if( connGhostRank[connIdx] < 0 )
    {
      for( localIndex const nbrIdx : connToLoc[connIdx] )
      {
        GEOSX_ERROR_IF_GE_MSG( numNeighbors, MAX_NEIGHBORS, "Too many neighbors, need to increase stack limit" );
        neighbors[numNeighbors++] = nbrIdx;
      }
    }
  }
  forUniqueValues( neighbors, numNeighbors, std::forward< LAMBDA >( lambda ) );
}

template< integer MAX_NEIGHBORS, typename NBR_MAP_TYPE, typename VAL_FUNC_TYPE, typename VAL_PRED_TYPE, typename LAMBDA >
void forUniqueNeighborValues( localIndex const locIdx,
                              NBR_MAP_TYPE const & neighbors,
                              VAL_FUNC_TYPE const & valueMap,
                              VAL_PRED_TYPE const & pred,
                              LAMBDA && lambda )
{
  using T = std::remove_cv_t< std::remove_reference_t< decltype( valueMap( localIndex {} ) ) >>;
  T nbrValues[MAX_NEIGHBORS];
  integer numValues = 0;
  for( localIndex const nbrIdx : neighbors[locIdx] )
  {
    GEOSX_ERROR_IF_GE_MSG( numValues, MAX_NEIGHBORS, "Too many neighbors, need to increase stack limit" );
    T const value = valueMap( nbrIdx );
    if( pred( value ) )
    {
      nbrValues[numValues++] = value;
    }
  }
  forUniqueValues( nbrValues, numValues, std::forward< LAMBDA >( lambda ) );
}

} // namespace meshUtils
} // namespace multiscale
} // namespace geosx

#endif //GEOSX_LINEARALGEBRA_MULTISCALE_MESHUTILS_HPP
