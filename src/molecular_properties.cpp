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

std::uint32_t implicit_hydrogen_count(const OEChem::OEAtomBase& atom) {
    // ``GetTotalHCount`` counts implicit hydrogens plus explicit hydrogen atoms
    // bonded to ``atom``. Weight and atom counts add this implicit portion (as
    // protium) while counting each explicit hydrogen atom separately, so an
    // explicit hydrogen — in particular an isotope such as ``[2H]`` that
    // ``OESuppressHydrogens`` retains — is counted exactly once with its own
    // mass rather than double-counted here and again as its own atom.
    std::uint32_t explicit_neighbors = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom.GetAtoms(); nbr; ++nbr) {
        if (is_hydrogen(*nbr)) {
            ++explicit_neighbors;
        }
    }
    const auto total = static_cast<std::uint32_t>(atom.GetTotalHCount());
    return total > explicit_neighbors ? total - explicit_neighbors : 0u;
}

double ExactMolecularWeight(const OEChem::OEMolBase& mol) {
    double exact_weight = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            exact_weight += atom_exact_mass(*atom);
            continue;
        }
        exact_weight += atom_exact_mass(*atom)
                        + static_cast<double>(implicit_hydrogen_count(*atom))
                              * default_isotopic_mass(1u);
    }
    return exact_weight;
}

double AverageMolecularWeight(const OEChem::OEMolBase& mol) {
    const auto total_atoms = TotalAtomCount(mol);
    return total_atoms == 0u
               ? 0.0
               : ExactMolecularWeight(mol) / static_cast<double>(total_atoms);
}

double StandardMolecularWeight(const OEChem::OEMolBase& mol) {
    // RDKit's MolWt sums standard average atomic weights per element and adds the
    // average mass of every bonded hydrogen. An explicit isotope uses that
    // isotope's exact mass instead of the element average, matching RDKit.
    const auto hydrogen_average_mass = OEChem::OEGetAverageWeight(1u);
    double standard_weight = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        const auto isotope = static_cast<std::uint32_t>(atom->GetIsotope());
        standard_weight += isotope != 0u
                               ? OEChem::OEGetIsotopicWeight(atomic_number, isotope)
                               : OEChem::OEGetAverageWeight(atomic_number);
        if (!is_hydrogen(*atom)) {
            standard_weight += static_cast<double>(implicit_hydrogen_count(*atom))
                               * hydrogen_average_mass;
        }
    }
    return standard_weight;
}

double heavy_atom_standard_weight(const OEChem::OEMolBase& mol) {
    // RDKit's HeavyAtomMolWt is the summed standard average weight of the heavy
    // atoms only, with all hydrogen mass removed. Summing the heavy atoms
    // directly is isotope-safe (an explicit isotope hydrogen contributes no
    // heavy mass) and byte-identical, on inputs without explicit hydrogen atoms,
    // to subtracting the hydrogen mass from the standard weight.
    double weight = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            continue;
        }
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        const auto isotope = static_cast<std::uint32_t>(atom->GetIsotope());
        weight += isotope != 0u
                      ? OEChem::OEGetIsotopicWeight(atomic_number, isotope)
                      : OEChem::OEGetAverageWeight(atomic_number);
    }
    return weight;
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
        hydrogens += implicit_hydrogen_count(*atom);
    }
    return heavy_atoms + hydrogens;
}

double RoundTopologicalPsa(double value) {
    return std::round(value * 100.0) / 100.0;
}

} // namespace OEFP
