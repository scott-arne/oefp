#ifndef OEFP_MOLECULAR_PROPERTIES_H
#define OEFP_MOLECULAR_PROPERTIES_H

#include <cstdint>

#include <oechem.h>

namespace OEFP {

/// \brief Return the exact (monoisotopic) molecular weight of a molecule.
///
/// The weight sums each atom's exact isotopic mass. For heavy atoms the
/// mass of their implicit and explicit hydrogens is added using the default
/// isotopic mass of hydrogen. Explicit hydrogen atoms contribute their own
/// exact isotopic mass. This reproduces Mordred's ``MW`` descriptor.
///
/// \param mol Molecule to weigh.
/// \returns Sum of per-atom exact masses including bonded hydrogens.
double ExactMolecularWeight(const OEChem::OEMolBase& mol);

/// \brief Return the average molecular weight per atom.
///
/// Computed as :cpp:func:`ExactMolecularWeight` divided by
/// :cpp:func:`TotalAtomCount` (heavy plus all hydrogens). This reproduces
/// Mordred's ``AMW`` descriptor.
///
/// \param mol Molecule to weigh.
/// \returns Average atomic weight, or 0.0 when the molecule has no atoms.
double AverageMolecularWeight(const OEChem::OEMolBase& mol);

/// \brief Return the number of heavy (non-hydrogen) atoms.
///
/// \param mol Molecule to count.
/// \returns Count of atoms whose atomic number is not hydrogen.
std::uint64_t HeavyAtomCount(const OEChem::OEMolBase& mol);

/// \brief Return the total atom count including hydrogens.
///
/// Counts heavy atoms plus their implicit and explicit hydrogens. This
/// reproduces Mordred's ``nAtom`` descriptor.
///
/// \param mol Molecule to count.
/// \returns Count of heavy atoms and all hydrogens.
std::uint64_t TotalAtomCount(const OEChem::OEMolBase& mol);

/// \brief Round a topological polar surface area value to two decimals.
///
/// Reproduces Mordred's ``TopoPSA`` rounding of ``std::round(value * 100) /
/// 100``. Sharing this helper keeps Mordred's ``TopoPSA`` and the OpenEye
/// ``TopologicalPSA`` column identical by construction.
///
/// \param value Raw polar surface area from ``OEGet2dPSA``.
/// \returns Value rounded to two decimal places.
double RoundTopologicalPsa(double value);

} // namespace OEFP

#endif // OEFP_MOLECULAR_PROPERTIES_H
