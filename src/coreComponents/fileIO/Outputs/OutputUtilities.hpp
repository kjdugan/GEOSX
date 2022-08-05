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
 * @file OutputUtilities.hpp
 */

#ifndef GEOSX_FILEIO_OUTPUTS_OUTPUTUTILITIES_HPP_
#define GEOSX_FILEIO_OUTPUTS_OUTPUTUTILITIES_HPP_

#include "common/DataTypes.hpp"
#include "dataRepository/RestartFlags.hpp"

#include <set>

namespace geosx
{

class ElementRegionManager;
class NodeManager;

namespace outputUtilities
{

/**
 * @brief Utility function that checks whether the field names provided by the user in `fieldNames` are actually registered somewhere.
 * @param[in] elemManager the elementRegionManager
 * @param[in] nodeManager the nodeManager
 * @param[in] fieldNames the set of field names specified by the user in the xml file
 * @param[in] outputName the output name (i.e., VTKOutput or SiloOutput)
 */
void checkFieldRegistration( ElementRegionManager const & elemManager,
                             NodeManager const & nodeManager,
                             std::set< string > const & fieldNames,
                             string const & outputName );

/**
 * @brief Utility function that checks whether a wrapper should be plotted or not
 * @param[in] wrapperPlotLevel the plot level of the wrapper
 * @param[in] requiredPlotLevel the plot level required by the user in the xml
 * @param[in] wrapperName the name of the wrapper
 * @param[in] fieldNames the set of field names specified by the user in the xml fiel
 * @param[in] onlyPlotSpecifiedFieldNames the flag that says whether we only plot the fields in fieldNames or not
 * @return true if the plot must be plotted, false otherwise
 */
bool isFieldPlotEnabled( dataRepository::PlotLevel const wrapperPlotLevel,
                         dataRepository::PlotLevel const requiredPlotLevel,
                         string const & wrapperName,
                         std::set< string > const & fieldNames,
                         integer const onlyPlotSpecifiedFieldNames );

} /* namespace outputUtilities */

} /* namespace geosx */

#endif /* GEOSX_FILEIO_OUTPUTS_OUTPUTILITIES_HPP_ */