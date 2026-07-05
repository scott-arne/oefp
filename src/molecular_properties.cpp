#include "oefp/molecular_properties.h"

#include <cmath>
#include <cstdint>

#include <oesystem.h>

namespace OEFP {

bool is_hydrogen(const OEChem::OEAtomBase& atom) {
    return atom.GetAtomicNum() == 1u;
}

double default_isotopic_mass(std::uint32_t atomic_number) {
    const auto mass_number = OEChem::OEGetDefaultMass(atomic_number);
    return OEChem::OEGetIsotopicWeight(atomic_number, mass_number);
}

double atom_exact_mass(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto isotope = static_cast<std::uint32_t>(atom.GetIsotope());
    if (isotope != 0u) {
        return OEChem::OEGetIsotopicWeight(atomic_number, isotope);
    }
    return default_isotopic_mass(atomic_number);
}

double ExactMolecularWeight(const OEChem::OEMolBase& mol) {
    double exact_weight = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            exact_weight += atom_exact_mass(*atom);
            continue;
        }
        const auto total_h_count = static_cast<std::uint32_t>(atom->GetTotalHCount());
        exact_weight += atom_exact_mass(*atom)
                        + static_cast<double>(total_h_count) * default_isotopic_mass(1u);
    }
    return exact_weight;
}

double AverageMolecularWeight(const OEChem::OEMolBase& mol) {
    const auto total_atoms = TotalAtomCount(mol);
    return total_atoms == 0u
               ? 0.0
               : ExactMolecularWeight(mol) / static_cast<double>(total_atoms);
}

std::uint64_t HeavyAtomCount(const OEChem::OEMolBase& mol) {
    std::uint64_t heavy_atoms = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!is_hydrogen(*atom)) {
            ++heavy_atoms;
        }
    }
    return heavy_atoms;
}

std::uint64_t TotalAtomCount(const OEChem::OEMolBase& mol) {
    std::uint64_t heavy_atoms = 0u;
    std::uint64_t hydrogens = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            ++hydrogens;
            continue;
        }
        ++heavy_atoms;
        hydrogens += static_cast<std::uint32_t>(atom->GetTotalHCount());
    }
    return heavy_atoms + hydrogens;
}

double RoundTopologicalPsa(double value) {
    return std::round(value * 100.0) / 100.0;
}

} // namespace OEFP
