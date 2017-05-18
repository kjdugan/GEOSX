/*
 * FiniteElementSpace.hpp
 *
 *  Created on: Aug 4, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CORE_SRC_FINITEELEMENT_FINITEELEMENTSPACE_HPP_
#define SRC_COMPONENTS_CORE_SRC_FINITEELEMENT_FINITEELEMENTSPACE_HPP_
#include "../dataRepository/ManagedGroup.hpp"

namespace geosx
{

class NodeManager;
class CellBlockManager;

namespace dataRepository
{
namespace keys
{
string const finiteElementSpace = "finiteElementSpace";
string const basis = "basis";
string const quadrature = "quadrature";
}
}

class FiniteElementSpace : public dataRepository::ManagedGroup
{
public:

  FiniteElementSpace() = delete;

  explicit FiniteElementSpace( std::string const & name, ManagedGroup * const parent );

  ~FiniteElementSpace();

  /**
   * @name Static Factory Catalog Functions
   */
  ///@{
  static string CatalogName() { return dataRepository::keys::finiteElementSpace; }

  ///@}

  virtual void BuildDataStructure( dataRepository::ManagedGroup * const parent );

  void FillDocumentationNode( dataRepository::ManagedGroup * const group );


  virtual dataRepository::ManagedGroup & getNodeManager();
  virtual dataRepository::ManagedGroup & getEdgeManager();
  virtual dataRepository::ManagedGroup & getFaceManager();
  virtual dataRepository::ManagedGroup & getElementManager();

private:

  NodeManager *    m_nodeManager    = nullptr;
  CellBlockManager * m_elementManager = nullptr;


};

} /* namespace geosx */

#endif /* SRC_COMPONENTS_CORE_SRC_FINITEELEMENT_FINITEELEMENTSPACE_HPP_ */
