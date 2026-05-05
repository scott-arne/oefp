#ifndef OEFP_H
#define OEFP_H

// Version information
#define OEFP_VERSION_MAJOR 0
#define OEFP_VERSION_MINOR 1
#define OEFP_VERSION_PATCH 0

#include "oefp/batch.h"
#include "oefp/compare.h"
#include "oefp/fingerprint.h"
#include "oefp/metric.h"

#include <oechem.h>

namespace OEFP {

/// \brief Calculate the molecular weight of a molecule.
///
/// Passes the molecule natively from Python to C++ via SWIG typemaps,
/// then delegates to OEChem's OECalculateMolecularWeight.
///
/// \param mol Reference to an OEMolBase object.
/// \returns Molecular weight in Daltons.
double calculate_molecular_weight(const OEChem::OEMolBase& mol);

} // namespace OEFP

#endif // OEFP_H
