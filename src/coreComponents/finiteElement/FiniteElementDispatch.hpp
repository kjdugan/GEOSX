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
 * @file FiniteElementDispatch.hpp
 */

#ifndef GEOSX_FINITEELEMENT_FINITEELEMENTDISPATCH_HPP_
#define GEOSX_FINITEELEMENT_FINITEELEMENTDISPATCH_HPP_

#include "elementFormulations/ConformingVirtualElementOrder1.hpp"
#include "elementFormulations/H1_Hexahedron_Lagrange1_GaussLegendre2.hpp"
#include "elementFormulations/H1_Pyramid_Lagrange1_Gauss5.hpp"
#include "elementFormulations/H1_Tetrahedron_Lagrange1_Gauss1.hpp"
#include "elementFormulations/H1_Wedge_Lagrange1_Gauss6.hpp"
#include "elementFormulations/Qk_Hexahedron_Lagrange_GaussLobatto.hpp"
#include "elementFormulations/H1_QuadrilateralFace_Lagrange1_GaussLegendre2.hpp"
#include "elementFormulations/H1_TriangleFace_Lagrange1_Gauss1.hpp"
#include "LvArray/src/system.hpp"

#define FE_1_TYPES \
  finiteElement::H1_Hexahedron_Lagrange1_GaussLegendre2, \
  finiteElement::H1_Wedge_Lagrange1_Gauss6, \
  finiteElement::H1_Tetrahedron_Lagrange1_Gauss1, \
  finiteElement::H1_Pyramid_Lagrange1_Gauss5
#define GL_FE_TYPES \
  finiteElement::Q1_Hexahedron_Lagrange_GaussLobatto, \
  finiteElement::Q2_Hexahedron_Lagrange_GaussLobatto, \
  finiteElement::Q3_Hexahedron_Lagrange_GaussLobatto, \
  finiteElement::Q4_Hexahedron_Lagrange_GaussLobatto, \
  finiteElement::Q5_Hexahedron_Lagrange_GaussLobatto
#ifdef GEOSX_DISPATCH_VEM
#define VEM_TYPES \
  finiteElement::H1_Tetrahedron_VEM_Gauss1, \
  finiteElement::H1_Wedge_VEM_Gauss1, \
  finiteElement::H1_Hexahedron_VEM_Gauss1, \
  finiteElement::H1_Prism5_VEM_Gauss1, \
  finiteElement::H1_Prism6_VEM_Gauss1, \
  finiteElement::H1_Prism7_VEM_Gauss1, \
  finiteElement::H1_Prism8_VEM_Gauss1, \
  finiteElement::H1_Prism9_VEM_Gauss1, \
  finiteElement::H1_Prism10_VEM_Gauss1, \
  finiteElement::H1_Prism11_VEM_Gauss1
#define BASE_FE_TYPES FE_1_TYPES, VEM_TYPES
#else
#define BASE_FE_TYPES FE_1_TYPES
#endif
#define ALL_FE_TYPES BASE_FE_TYPES, GL_FE_TYPES

#define FE_TYPES_2D \
  finiteElement::H1_QuadrilateralFace_Lagrange1_GaussLegendre2.hpp  \
  finiteElement::H1_TriangleFace_Lagrange1_Gauss1.hpp

namespace geosx
{
namespace finiteElement
{

/**
 * @brief Helper structure for dynamic finite element dispatch
 * @tparam FE_TYPES list of finite element types to handle
 */
template< typename ... FE_TYPES >
struct FiniteElementDispatchHandler {};

/**
 * @brief Finite elemnt dispatch fall-through in case all casts were unsuccessful: always error
 */
template<>
struct FiniteElementDispatchHandler<>
{
  template< typename LAMBDA >
  static void
  dispatch3D( FiniteElementBase const & input,
              LAMBDA && GEOSX_UNUSED_PARAM( lambda ) )
  {
    GEOSX_ERROR( "finiteElement::dispatch3D() is not implemented for input of "<<typeid(input).name() );
  }

  template< typename LAMBDA >
  static void
  dispatch3D( FiniteElementBase & input,
              LAMBDA && GEOSX_UNUSED_PARAM( lambda ) )
  {
    GEOSX_ERROR( "finiteElement::dispatch3D() is not implemented for input of "<<typeid(input).name() );
  }

  template< typename LAMBDA >
  static void
  dispatch2D( FiniteElementBase const & input,
              LAMBDA && GEOSX_UNUSED_PARAM( lambda ) )
  {
    GEOSX_ERROR( "finiteElement::dispatch2D() is not implemented for input of: "<<LvArray::system::demangleType( &input ) );
  }
};

/**
 * @brief Structure for recursive finite element dispatch implementation
 * @tparam FE_TYPE first finite element type to handle
 * @tparam FE_TYPES following finite element types to handle
 */
template< typename FE_TYPE, typename ... FE_TYPES >
struct FiniteElementDispatchHandler< FE_TYPE, FE_TYPES... >
{
  template< typename LAMBDA >
  static void
  dispatch3D( FiniteElementBase const & input,
              LAMBDA && lambda )
  {
    if( auto const * const ptr = dynamic_cast< FE_TYPE const * >(&input) )
    {
      lambda( *ptr );
    }
    else
    {
      FiniteElementDispatchHandler< FE_TYPES... >::dispatch3D( input, lambda );
    }
  }

  template< typename LAMBDA >
  static void
  dispatch3D( FiniteElementBase & input,
              LAMBDA && lambda )
  {
    if( auto * const ptr = dynamic_cast< FE_TYPE * >(&input) )
    {
      lambda( *ptr );
    }
    else
    {
      FiniteElementDispatchHandler< FE_TYPES... >::dispatch3D( input, lambda );
    }
  }

  template< typename LAMBDA >
  static void
  dispatch2D( FiniteElementBase const & input,
              LAMBDA && lambda )
  {
    if( auto const * const ptr = dynamic_cast< FE_TYPE const * >(&input) )
    {
      lambda( *ptr );
    }
    else
    {
      FiniteElementDispatchHandler< FE_TYPES... >::dispatch2D( input, lambda );
    }
  }
};



template< typename LAMBDA >
void
dispatchlowOrder3D( FiniteElementBase const & input,
                    LAMBDA && lambda )
{
  if( auto const * const ptr1 = dynamic_cast< H1_Hexahedron_Lagrange1_GaussLegendre2 const * >(&input) )
  {
    lambda( *ptr1 );
  }
  else if( auto const * const ptr2 = dynamic_cast< H1_Wedge_Lagrange1_Gauss6 const * >(&input) )
  {
    lambda( *ptr2 );
  }
  else if( auto const * const ptr3 = dynamic_cast< H1_Tetrahedron_Lagrange1_Gauss1 const * >(&input) )
  {
    lambda( *ptr3 );
  }
  else if( auto const * const ptr4 = dynamic_cast< H1_Pyramid_Lagrange1_Gauss5 const * >(&input) )
  {
    lambda( *ptr4 );
  }
  else
  {
    GEOSX_ERROR( "finiteElement::dispatchlowOrder3D() is not implemented for input of "<<LvArray::system::demangleType( &input ) );
  }
}

} // namespace finiteElement

} // namespace geosx



#endif /* GEOSX_FINITEELEMENT_FINITEELEMENTDISPATCH_HPP_ */
