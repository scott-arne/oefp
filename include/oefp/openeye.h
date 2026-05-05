#ifndef OEFP_OPENEYE_H
#define OEFP_OPENEYE_H

#include "oefp/fingerprint.h"

#include <oegraphsim.h>

namespace OEFP {

/// \brief Convert an OpenEye fingerprint into OEFP dense binary storage.
OEFP FromOEFingerPrint(const OEGraphSim::OEFingerPrint& fp);

/// \brief Convert an OEFP dense binary fingerprint back to OEFingerPrint.
///
/// \raises std::invalid_argument: When the OEFP spec does not carry a
///     resolvable OpenEye fingerprint type.
OEGraphSim::OEFingerPrint ToOEFingerPrint(const OEFP& fp);

} // namespace OEFP

#endif // OEFP_OPENEYE_H
