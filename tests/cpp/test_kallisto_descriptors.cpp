#include "oefp/kallisto_descriptors.h"
#include "oefp/kallisto_geometry.h"
#include "oefp/kallisto_data.h"
#include "oefp/atom_descriptor.h"

#include <gtest/gtest.h>
#include <oechem.h>
#include <cmath>
#include <vector>
#include <utility>

namespace OEFP {
namespace {

// Test coordination_numbers with cntype "erf" on H2
TEST(KallistoDescriptors, CoordinationNumbersErfH2) {
    // H2 molecule: H at (0,0,0), H at (1.4,0,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    // Set coordinates in Angstrom (context converts to Bohr)
    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto cn_erf = coordination_numbers(ctx, "erf");
    ASSERT_EQ(cn_erf.size(), 2u);
    EXPECT_NEAR(cn_erf[0], 0.91896554, 1e-6);
    EXPECT_NEAR(cn_erf[1], 0.91896554, 1e-6);
}

// Test coordination_numbers with cntype "cov" on H2
TEST(KallistoDescriptors, CoordinationNumbersCovH2) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto cn_cov = coordination_numbers(ctx, "cov");
    ASSERT_EQ(cn_cov.size(), 2u);
    EXPECT_NEAR(cn_cov[0], 0.90137656, 1e-6);
    EXPECT_NEAR(cn_cov[1], 0.90137656, 1e-6);
}

// Test coordination_numbers with cntype "exp" on H2
TEST(KallistoDescriptors, CoordinationNumbersExpH2) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto cn_exp = coordination_numbers(ctx, "exp");
    ASSERT_EQ(cn_exp.size(), 2u);
    EXPECT_NEAR(cn_exp[0], 0.91903651, 1e-6);
    EXPECT_NEAR(cn_exp[1], 0.91903651, 1e-6);
}

// Test proximity_shells on H2
TEST(KallistoDescriptors, ProximityShellsH2) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto prox = proximity_shells(ctx, {2, 3});
    ASSERT_EQ(prox.size(), 2u);
    EXPECT_NEAR(prox[0], 9.53630841e-10, 1e-6);
    EXPECT_NEAR(prox[1], 9.53630841e-10, 1e-6);
}

// Test coordination_numbers on H2O triatomic
TEST(KallistoDescriptors, CoordinationNumbersH2O) {
    // O at (0,0,0), H at (1.8,0,0), H at (0,1.8,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);   // Oxygen
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Test all three CN types
    auto cn_erf = coordination_numbers(ctx, "erf");
    ASSERT_EQ(cn_erf.size(), 3u);
    EXPECT_NEAR(cn_erf[0], 1.99147549, 1e-6);
    EXPECT_NEAR(cn_erf[1], 0.99573774, 1e-6);
    EXPECT_NEAR(cn_erf[2], 0.99573774, 1e-6);

    auto cn_cov = coordination_numbers(ctx, "cov");
    ASSERT_EQ(cn_cov.size(), 3u);
    EXPECT_NEAR(cn_cov[0], 1.61210366, 1e-6);
    EXPECT_NEAR(cn_cov[1], 0.80605183, 1e-6);
    EXPECT_NEAR(cn_cov[2], 0.80605183, 1e-6);

    auto cn_exp = coordination_numbers(ctx, "exp");
    ASSERT_EQ(cn_exp.size(), 3u);
    EXPECT_NEAR(cn_exp[0], 1.98983559, 1e-6);
    EXPECT_NEAR(cn_exp[1], 0.99774852, 1e-6);
    EXPECT_NEAR(cn_exp[2], 0.99774852, 1e-6);
}

// Test proximity_shells on H2O
TEST(KallistoDescriptors, ProximityShellsH2O) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto prox = proximity_shells(ctx, {2, 3});
    ASSERT_EQ(prox.size(), 3u);
    EXPECT_NEAR(prox[0], 2.93567393e-11, 1e-6);
    EXPECT_NEAR(prox[1], 1.24699443e-02, 1e-6);
    EXPECT_NEAR(prox[2], 1.24699443e-02, 1e-6);
}

// Test that CovalentCoordinationNumbers() returns the same as coordination_numbers(..., "cov")
TEST(KallistoDescriptors, CovalentCoordinationNumbersCaching) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Get covalent CN via the kernel
    auto cn_cov_kernel = coordination_numbers(ctx, "cov");

    // Get covalent CN via the cached accessor
    const auto& cn_cov_cached = ctx.CovalentCoordinationNumbers();

    // They should match
    ASSERT_EQ(cn_cov_kernel.size(), cn_cov_cached.size());
    for (std::size_t i = 0; i < cn_cov_kernel.size(); ++i) {
        EXPECT_NEAR(cn_cov_kernel[i], cn_cov_cached[i], 1e-12);
    }

    // Call again to verify caching works
    const auto& cn_cov_cached2 = ctx.CovalentCoordinationNumbers();
    ASSERT_EQ(cn_cov_cached.size(), cn_cov_cached2.size());
    for (std::size_t i = 0; i < cn_cov_cached.size(); ++i) {
        EXPECT_EQ(cn_cov_cached[i], cn_cov_cached2[i]);
    }
}

// Test KallistoAtomDescriptorSchema
TEST(KallistoDescriptors, AtomDescriptorSchema) {
    auto schema = KallistoAtomDescriptorSchema();
    ASSERT_NE(schema, nullptr);

    // Should have 8 columns for Task 8
    ASSERT_EQ(schema->Size(), 8u);

    // Check each column
    EXPECT_EQ(schema->Definition(0).name, "cn_erf");
    EXPECT_EQ(schema->Definition(0).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(0).group, "kallisto");
    EXPECT_EQ(schema->Definition(0).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(0).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(1).name, "cn_cov");
    EXPECT_EQ(schema->Definition(1).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(1).group, "kallisto");
    EXPECT_EQ(schema->Definition(1).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(1).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(2).name, "cn_exp");
    EXPECT_EQ(schema->Definition(2).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(2).group, "kallisto");
    EXPECT_EQ(schema->Definition(2).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(2).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(3).name, "prox");
    EXPECT_EQ(schema->Definition(3).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(3).group, "kallisto");
    EXPECT_EQ(schema->Definition(3).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(3).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(4).name, "eeq");
    EXPECT_EQ(schema->Definition(4).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(4).group, "kallisto");
    EXPECT_EQ(schema->Definition(4).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(4).units, "e");
    EXPECT_EQ(schema->Definition(4).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(5).name, "alp");
    EXPECT_EQ(schema->Definition(5).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(5).group, "kallisto");
    EXPECT_EQ(schema->Definition(5).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(5).units, "Bohr^3");
    EXPECT_EQ(schema->Definition(5).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(6).name, "vdw_rahm");
    EXPECT_EQ(schema->Definition(6).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(6).group, "kallisto");
    EXPECT_EQ(schema->Definition(6).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(6).units, "Bohr");
    EXPECT_EQ(schema->Definition(6).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(7).name, "vdw_truhlar");
    EXPECT_EQ(schema->Definition(7).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(7).group, "kallisto");
    EXPECT_EQ(schema->Definition(7).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(7).units, "Bohr");
    EXPECT_EQ(schema->Definition(7).prerequisites, kDescriptorPrerequisiteCoordinates3D);
}

// Test that kernels return empty vectors for ineligible contexts (2D molecule)
TEST(KallistoDescriptors, IneligibleContext2D) {
    // Create a 2D molecule (no 3D coords)
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c1 = mol.NewAtom(6);  // Carbon
    OEChem::OEAtomBase* c2 = mol.NewAtom(6);
    mol.NewBond(c1, c2, 1);

    // Set 2D coords to ensure dimension is 2
    const double coords_c1[3] = {0.0, 0.0, 0.0};
    const double coords_c2[3] = {1.5, 0.0, 0.0};
    mol.SetCoords(c1, coords_c1);
    mol.SetCoords(c2, coords_c2);
    mol.SetDimension(2);

    ASSERT_EQ(mol.GetDimension(), 2);

    KallistoGeometryContext ctx(mol);
    EXPECT_FALSE(ctx.Eligible());
    EXPECT_EQ(ctx.AtomCount(), 2u);
    EXPECT_EQ(ctx.AtomicNumbers().size(), 0u);
    EXPECT_EQ(ctx.CoordsBohr().size(), 0u);

    // All kernels should return empty
    auto cn_erf = coordination_numbers(ctx, "erf");
    EXPECT_EQ(cn_erf.size(), 0u);

    auto cn_cov = coordination_numbers(ctx, "cov");
    EXPECT_EQ(cn_cov.size(), 0u);

    auto cn_exp = coordination_numbers(ctx, "exp");
    EXPECT_EQ(cn_exp.size(), 0u);

    auto prox = proximity_shells(ctx, {2, 3});
    EXPECT_EQ(prox.size(), 0u);

    // Cached covalent CN should also return empty
    const auto& cn_cov_cached = ctx.CovalentCoordinationNumbers();
    EXPECT_EQ(cn_cov_cached.size(), 0u);
}

// Test that kernels return empty vectors for ineligible contexts (Z > 86)
TEST(KallistoDescriptors, IneligibleContextHeavyElement) {
    // Create a molecule with Z > 86 (radon is 86, francium is 87)
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* fr = mol.NewAtom(87);  // Francium, Z=87 > 86
    mol.SetDimension(3);

    constexpr double coords[3] = {0.0, 0.0, 0.0};
    mol.SetCoords(fr, coords);

    KallistoGeometryContext ctx(mol);
    EXPECT_FALSE(ctx.Eligible());
    EXPECT_EQ(ctx.AtomCount(), 1u);
    EXPECT_EQ(ctx.AtomicNumbers().size(), 0u);
    EXPECT_EQ(ctx.CoordsBohr().size(), 0u);

    // All kernels should return empty
    auto cn_erf = coordination_numbers(ctx, "erf");
    EXPECT_EQ(cn_erf.size(), 0u);

    auto cn_cov = coordination_numbers(ctx, "cov");
    EXPECT_EQ(cn_cov.size(), 0u);

    auto cn_exp = coordination_numbers(ctx, "exp");
    EXPECT_EQ(cn_exp.size(), 0u);

    auto prox = proximity_shells(ctx, {2, 3});
    EXPECT_EQ(prox.size(), 0u);

    // Cached covalent CN should also return empty
    const auto& cn_cov_cached = ctx.CovalentCoordinationNumbers();
    EXPECT_EQ(cn_cov_cached.size(), 0u);
}

// Test that atom descriptor rows preserve non-contiguous OpenEye atom IDs
TEST(KallistoDescriptors, NonContiguousAtomIndices) {
    // Build a molecule with 5 atoms, set 3D coords, then delete the middle atom
    // to create non-contiguous atom indices
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c0 = mol.NewAtom(6);  // Carbon
    OEChem::OEAtomBase* c1 = mol.NewAtom(6);
    OEChem::OEAtomBase* c2 = mol.NewAtom(6);
    OEChem::OEAtomBase* c3 = mol.NewAtom(6);
    OEChem::OEAtomBase* c4 = mol.NewAtom(6);

    // Set 3D coordinates (arbitrary positions)
    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_0[3] = {0.0, 0.0, 0.0};
    const double coords_1[3] = {1.5, 0.0, 0.0};
    const double coords_2[3] = {3.0, 0.0, 0.0};
    const double coords_3[3] = {4.5, 0.0, 0.0};
    const double coords_4[3] = {6.0, 0.0, 0.0};
    mol.SetCoords(c0, coords_0);
    mol.SetCoords(c1, coords_1);
    mol.SetCoords(c2, coords_2);
    mol.SetCoords(c3, coords_3);
    mol.SetCoords(c4, coords_4);
    mol.SetDimension(3);

    // Verify initial indices are contiguous: 0,1,2,3,4
    EXPECT_EQ(c0->GetIdx(), 0u);
    EXPECT_EQ(c1->GetIdx(), 1u);
    EXPECT_EQ(c2->GetIdx(), 2u);
    EXPECT_EQ(c3->GetIdx(), 3u);
    EXPECT_EQ(c4->GetIdx(), 4u);

    // Delete the middle atom (c2, idx=2)
    mol.DeleteAtom(c2);

    // Collect remaining atom indices via iteration (OpenEye order)
    std::vector<unsigned int> expected_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        expected_indices.push_back(atom->GetIdx());
    }

    // Verify the remaining indices are non-contiguous (e.g., {0,1,3,4})
    ASSERT_EQ(expected_indices.size(), 4u);
    EXPECT_EQ(expected_indices[0], 0u);
    EXPECT_EQ(expected_indices[1], 1u);
    EXPECT_EQ(expected_indices[2], 3u);  // Not 2!
    EXPECT_EQ(expected_indices[3], 4u);  // Not 3!

    // Build geometry context
    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());
    ASSERT_EQ(ctx.AtomCount(), 4u);

    // Verify context AtomIndices() matches the non-contiguous IDs
    const auto& ctx_indices = ctx.AtomIndices();
    ASSERT_EQ(ctx_indices.size(), 4u);
    EXPECT_EQ(ctx_indices[0], 0u);
    EXPECT_EQ(ctx_indices[1], 1u);
    EXPECT_EQ(ctx_indices[2], 3u);
    EXPECT_EQ(ctx_indices[3], 4u);

    // Compute atom descriptors
    auto result = MakeKallistoAtomDescriptors(mol);
    ASSERT_EQ(result.AtomCount(), 4u);

    // Verify the descriptor set's AtomIndices() matches the non-contiguous pattern
    const auto& result_indices = result.AtomIndices();
    ASSERT_EQ(result_indices.size(), 4u);
    EXPECT_EQ(result_indices[0], 0u);
    EXPECT_EQ(result_indices[1], 1u);
    EXPECT_EQ(result_indices[2], 3u);
    EXPECT_EQ(result_indices[3], 4u);
}

// Test eeq_charges on water: charges sum to total charge and match kallisto values
TEST(KallistoDescriptors, EeqChargesH2O) {
    // H2O: O at (0,0,0), H at (1.8,0,0), H at (0,1.8,0) Bohr (same coords as CN test)
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);   // Oxygen
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Get EEQ charges from the kernel
    auto eeq = eeq_charges(ctx);
    ASSERT_EQ(eeq.size(), 3u);

    // Constraint: charges sum to total charge (neutral molecule = 0)
    double charge_sum = eeq[0] + eeq[1] + eeq[2];
    EXPECT_NEAR(charge_sum, 0.0, 1e-9);

    // Expected values from kallisto 1.0.10 on same Bohr coords:
    // python -c "from kallisto.molecule import Molecule; import numpy as np;
    // m = Molecule(numbers=[8,1,1], positions=np.array([[0,0,0],[1.8,0,0],[0,1.8,0]]));
    // print(m.get_eeq(0))"
    // [-0.60062105  0.30031053  0.30031053]
    EXPECT_NEAR(eeq[0], -0.60062105, 1e-6);
    EXPECT_NEAR(eeq[1],  0.30031053, 1e-6);
    EXPECT_NEAR(eeq[2],  0.30031053, 1e-6);
}

// Test that EeqCharges() cached accessor returns same as eeq_charges() kernel
TEST(KallistoDescriptors, EeqChargesCaching) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Get EEQ via the kernel
    auto eeq_kernel = eeq_charges(ctx);

    // Get EEQ via the cached accessor
    const auto& eeq_cached = ctx.EeqCharges();

    // They should match
    ASSERT_EQ(eeq_kernel.size(), eeq_cached.size());
    for (std::size_t i = 0; i < eeq_kernel.size(); ++i) {
        EXPECT_NEAR(eeq_kernel[i], eeq_cached[i], 1e-12);
    }

    // Call again to verify caching works
    const auto& eeq_cached2 = ctx.EeqCharges();
    ASSERT_EQ(eeq_cached.size(), eeq_cached2.size());
    for (std::size_t i = 0; i < eeq_cached.size(); ++i) {
        EXPECT_EQ(eeq_cached[i], eeq_cached2[i]);
    }
}

// Test polarizabilities on water: verify against kallisto 1.0.10 values
TEST(KallistoDescriptors, PolarizabilitiesH2O) {
    // H2O: O at (0,0,0), H at (1.8,0,0), H at (0,1.8,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Get polarizabilities from the kernel
    auto alp = polarizabilities(ctx);
    ASSERT_EQ(alp.size(), 3u);

    // Expected values from kallisto 1.0.10 on same Bohr coords:
    // [6.76353129 1.33254911 1.33254911]
    EXPECT_NEAR(alp[0], 6.76353129, 1e-5);
    EXPECT_NEAR(alp[1], 1.33254911, 1e-5);
    EXPECT_NEAR(alp[2], 1.33254911, 1e-5);
}

// Test polarizabilities on methane
TEST(KallistoDescriptors, PolarizabilitiesCH4) {
    // CH4: C at (0,0,0), H at various positions
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c = mol.NewAtom(6);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    OEChem::OEAtomBase* h3 = mol.NewAtom(1);
    OEChem::OEAtomBase* h4 = mol.NewAtom(1);
    mol.NewBond(c, h1, 1);
    mol.NewBond(c, h2, 1);
    mol.NewBond(c, h3, 1);
    mol.NewBond(c, h4, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_c[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {2.05 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 2.05 * bohr_to_angstrom, 0.0};
    const double coords_h3[3] = {0.0, 0.0, 2.05 * bohr_to_angstrom};
    const double coords_h4[3] = {-1.5 * bohr_to_angstrom, -1.5 * bohr_to_angstrom, -1.5 * bohr_to_angstrom};
    mol.SetCoords(c, coords_c);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetCoords(h3, coords_h3);
    mol.SetCoords(h4, coords_h4);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto alp = polarizabilities(ctx);
    ASSERT_EQ(alp.size(), 5u);

    // Expected from kallisto: [9.60182599 2.17481057 2.17481057 2.17481057 2.56391108]
    EXPECT_NEAR(alp[0], 9.60182599, 1e-5);
    EXPECT_NEAR(alp[1], 2.17481057, 1e-5);
    EXPECT_NEAR(alp[2], 2.17481057, 1e-5);
    EXPECT_NEAR(alp[3], 2.17481057, 1e-5);
    EXPECT_NEAR(alp[4], 2.56391108, 1e-5);
}

// Test polarizabilities on ammonia (heteroatom)
TEST(KallistoDescriptors, PolarizabilitiesNH3) {
    // NH3: N at (0,0,0), H at various positions
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* n = mol.NewAtom(7);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    OEChem::OEAtomBase* h3 = mol.NewAtom(1);
    mol.NewBond(n, h1, 1);
    mol.NewBond(n, h2, 1);
    mol.NewBond(n, h3, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_n[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.9 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.9 * bohr_to_angstrom, 0.0};
    const double coords_h3[3] = {0.0, 0.0, 1.9 * bohr_to_angstrom};
    mol.SetCoords(n, coords_n);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetCoords(h3, coords_h3);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto alp = polarizabilities(ctx);
    ASSERT_EQ(alp.size(), 4u);

    // Expected from kallisto: [9.52888686 1.44685267 1.44685267 1.44685267]
    EXPECT_NEAR(alp[0], 9.52888686, 1e-5);
    EXPECT_NEAR(alp[1], 1.44685267, 1e-5);
    EXPECT_NEAR(alp[2], 1.44685267, 1e-5);
    EXPECT_NEAR(alp[3], 1.44685267, 1e-5);
}

// Test van_der_waals_radii with vdwtype "rahm" on H2O
TEST(KallistoDescriptors, VanDerWaalsRadiiRahmH2O) {
    // H2O molecule: O at origin, H at (1.8,0,0), H at (0,1.8,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);   // Oxygen
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto vdw_rahm = van_der_waals_radii(ctx, "rahm");
    ASSERT_EQ(vdw_rahm.size(), 3u);

    // Expected from kallisto: [3.33756653 2.4081692  2.4081692]
    EXPECT_NEAR(vdw_rahm[0], 3.33756653, 1e-5);
    EXPECT_NEAR(vdw_rahm[1], 2.4081692, 1e-5);
    EXPECT_NEAR(vdw_rahm[2], 2.4081692, 1e-5);
}

// Test van_der_waals_radii with vdwtype "truhlar" on H2O
TEST(KallistoDescriptors, VanDerWaalsRadiiTruhlarH2O) {
    // H2O molecule: O at origin, H at (1.8,0,0), H at (0,1.8,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);   // Oxygen
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto vdw_truhlar = van_der_waals_radii(ctx, "truhlar");
    ASSERT_EQ(vdw_truhlar.size(), 3u);

    // Expected from kallisto: [2.97043421 1.72012086 1.72012086]
    EXPECT_NEAR(vdw_truhlar[0], 2.97043421, 1e-5);
    EXPECT_NEAR(vdw_truhlar[1], 1.72012086, 1e-5);
    EXPECT_NEAR(vdw_truhlar[2], 1.72012086, 1e-5);
}

// Test that MakeKallistoAtomDescriptors emits mutually consistent alp/vdw columns.
// This guards against future duplication of the polarizability kernel: the columns
// can only be consistent if they derive from the SAME underlying polarizability vector.
TEST(KallistoDescriptors, AlpVdwConsistencyInEmission) {
    // CH4: multi-atom molecule with varying atomic numbers
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c = mol.NewAtom(6);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    OEChem::OEAtomBase* h3 = mol.NewAtom(1);
    OEChem::OEAtomBase* h4 = mol.NewAtom(1);
    mol.NewBond(c, h1, 1);
    mol.NewBond(c, h2, 1);
    mol.NewBond(c, h3, 1);
    mol.NewBond(c, h4, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_c[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {2.05 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 2.05 * bohr_to_angstrom, 0.0};
    const double coords_h3[3] = {0.0, 0.0, 2.05 * bohr_to_angstrom};
    const double coords_h4[3] = {-1.5 * bohr_to_angstrom, -1.5 * bohr_to_angstrom, -1.5 * bohr_to_angstrom};
    mol.SetCoords(c, coords_c);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetCoords(h3, coords_h3);
    mol.SetCoords(h4, coords_h4);
    mol.SetDimension(3);

    // Emit the full descriptor set
    auto descriptors = MakeKallistoAtomDescriptors(mol);
    ASSERT_EQ(descriptors.AtomCount(), 5u);

    // Consistency check: for each atom i, verify
    //   vdw_rahm[i] == VDW_RAHM[Z_i] * 2.54 * alp[i]^(1/7)
    //   vdw_truhlar[i] == VDW_TRUHLAR[Z_i] * 2.54 * alp[i]^(1/7)
    // to high precision (1e-12).
    // This only holds if the same polarizability vector underlies both columns.
    constexpr double theta_a = 2.54;
    constexpr double osev = 1.0 / 7.0;

    // Column indices: alp=5, vdw_rahm=6, vdw_truhlar=7
    constexpr std::size_t alp_col = 5;
    constexpr std::size_t vdw_rahm_col = 6;
    constexpr std::size_t vdw_truhlar_col = 7;

    // Atomic numbers for CH4: C=6, H=1 (4 times)
    const int zs[5] = {6, 1, 1, 1, 1};

    for (std::size_t i = 0; i < 5; ++i) {
        const auto alp_val = descriptors.Value(i, alp_col);
        const auto vdw_rahm_val = descriptors.Value(i, vdw_rahm_col);
        const auto vdw_truhlar_val = descriptors.Value(i, vdw_truhlar_col);

        ASSERT_TRUE(alp_val.has_value()) << "alp missing at atom " << i;
        ASSERT_TRUE(vdw_rahm_val.has_value()) << "vdw_rahm missing at atom " << i;
        ASSERT_TRUE(vdw_truhlar_val.has_value()) << "vdw_truhlar missing at atom " << i;

        const double alp = alp_val.value();
        const double vdw_rahm = vdw_rahm_val.value();
        const double vdw_truhlar = vdw_truhlar_val.value();

        const int zi = zs[i];
        const double theta_rahm = kallisto::VDW_RAHM[zi];
        const double theta_truhlar = kallisto::VDW_TRUHLAR[zi];

        const double expected_rahm = theta_rahm * theta_a * std::pow(alp, osev);
        const double expected_truhlar = theta_truhlar * theta_a * std::pow(alp, osev);

        EXPECT_NEAR(vdw_rahm, expected_rahm, 1e-12)
            << "vdw_rahm inconsistent with alp at atom " << i;
        EXPECT_NEAR(vdw_truhlar, expected_truhlar, 1e-12)
            << "vdw_truhlar inconsistent with alp at atom " << i;
    }
}

// Test KallistoSterimol on CH4 with fixed geometry
TEST(KallistoDescriptors, SterimolCH4) {
    // CH4: C at origin, 4 H in tetrahedral geometry (Bohr coords)
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c = mol.NewAtom(6);   // Carbon
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    OEChem::OEAtomBase* h3 = mol.NewAtom(1);
    OEChem::OEAtomBase* h4 = mol.NewAtom(1);

    mol.NewBond(c, h1, 1);
    mol.NewBond(c, h2, 1);
    mol.NewBond(c, h3, 1);
    mol.NewBond(c, h4, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_c[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.2 * bohr_to_angstrom, 1.2 * bohr_to_angstrom, 1.2 * bohr_to_angstrom};
    const double coords_h2[3] = {-1.2 * bohr_to_angstrom, -1.2 * bohr_to_angstrom, 1.2 * bohr_to_angstrom};
    const double coords_h3[3] = {-1.2 * bohr_to_angstrom, 1.2 * bohr_to_angstrom, -1.2 * bohr_to_angstrom};
    const double coords_h4[3] = {1.2 * bohr_to_angstrom, -1.2 * bohr_to_angstrom, -1.2 * bohr_to_angstrom};

    mol.SetCoords(c, coords_c);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetCoords(h3, coords_h3);
    mol.SetCoords(h4, coords_h4);
    mol.SetDimension(3);

    // Test both directions for C-H1 bond (positional indices 0,1 in the context)
    // Expected values from kallisto 1.0.10:
    // CH4 C(0)->H1(1): L=4.6605247297, B1=3.5655700256, B5=4.5416550338
    // CH4 H1(1)->C(0): L=5.4742409055, B1=3.5630969690, B5=4.5416508654

    auto s1 = KallistoSterimol(mol, 0, 1);  // C -> H1
    ASSERT_TRUE(s1.has_value());
    EXPECT_NEAR(s1->l, 4.6605247297, 1e-5);
    EXPECT_NEAR(s1->b1, 3.5655700256, 1e-5);
    EXPECT_NEAR(s1->b5, 4.5416550338, 1e-5);

    auto s2 = KallistoSterimol(mol, 1, 0);  // H1 -> C
    ASSERT_TRUE(s2.has_value());
    EXPECT_NEAR(s2->l, 5.4742409055, 1e-5);
    EXPECT_NEAR(s2->b1, 3.5630969690, 1e-5);
    EXPECT_NEAR(s2->b5, 4.5416508654, 1e-5);
}

// Test MakeKallistoBondDescriptors on a toluene-like molecule with ring + acyclic bonds
TEST(KallistoDescriptors, BondDescriptorsToluene) {
    // Simplified toluene: benzene ring (6 carbons) + methyl group (C + 3 H)
    OEChem::OEGraphMol mol;

    // Ring carbons
    OEChem::OEAtomBase* c0 = mol.NewAtom(6);
    OEChem::OEAtomBase* c1 = mol.NewAtom(6);
    OEChem::OEAtomBase* c2 = mol.NewAtom(6);
    OEChem::OEAtomBase* c3 = mol.NewAtom(6);
    OEChem::OEAtomBase* c4 = mol.NewAtom(6);
    OEChem::OEAtomBase* c5 = mol.NewAtom(6);

    // Methyl group
    OEChem::OEAtomBase* c6 = mol.NewAtom(6);  // methyl carbon
    OEChem::OEAtomBase* h7 = mol.NewAtom(1);
    OEChem::OEAtomBase* h8 = mol.NewAtom(1);
    OEChem::OEAtomBase* h9 = mol.NewAtom(1);

    // Ring bonds
    mol.NewBond(c0, c1, 1);
    mol.NewBond(c1, c2, 1);
    mol.NewBond(c2, c3, 1);
    mol.NewBond(c3, c4, 1);
    mol.NewBond(c4, c5, 1);
    mol.NewBond(c5, c0, 1);

    // Acyclic bonds (ring to methyl, methyl to H)
    mol.NewBond(c0, c6, 1);
    mol.NewBond(c6, h7, 1);
    mol.NewBond(c6, h8, 1);
    mol.NewBond(c6, h9, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;

    // Ring coords (xy-plane)
    const double coords_c0[3] = {2.65 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_c1[3] = {1.32 * bohr_to_angstrom, 2.29 * bohr_to_angstrom, 0.0};
    const double coords_c2[3] = {-1.32 * bohr_to_angstrom, 2.29 * bohr_to_angstrom, 0.0};
    const double coords_c3[3] = {-2.65 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_c4[3] = {-1.32 * bohr_to_angstrom, -2.29 * bohr_to_angstrom, 0.0};
    const double coords_c5[3] = {1.32 * bohr_to_angstrom, -2.29 * bohr_to_angstrom, 0.0};

    // Methyl coords (extending along x)
    const double coords_c6[3] = {5.0 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h7[3] = {5.8 * bohr_to_angstrom, 1.5 * bohr_to_angstrom, 0.8 * bohr_to_angstrom};
    const double coords_h8[3] = {5.8 * bohr_to_angstrom, -1.5 * bohr_to_angstrom, 0.8 * bohr_to_angstrom};
    const double coords_h9[3] = {5.8 * bohr_to_angstrom, 0.0, -1.6 * bohr_to_angstrom};

    mol.SetCoords(c0, coords_c0);
    mol.SetCoords(c1, coords_c1);
    mol.SetCoords(c2, coords_c2);
    mol.SetCoords(c3, coords_c3);
    mol.SetCoords(c4, coords_c4);
    mol.SetCoords(c5, coords_c5);
    mol.SetCoords(c6, coords_c6);
    mol.SetCoords(h7, coords_h7);
    mol.SetCoords(h8, coords_h8);
    mol.SetCoords(h9, coords_h9);
    mol.SetDimension(3);

    // Compute bond descriptors
    auto bond_descs = MakeKallistoBondDescriptors(mol);

    // Expected: 8 acyclic directed bonds (both directions):
    // c0-c6, c6-c0, c6-h7, h7-c6, c6-h8, h8-c6, c6-h9, h9-c6
    // Ring bonds should be EXCLUDED
    ASSERT_EQ(bond_descs.BondCount(), 8u);

    // Verify schema
    const auto& schema = bond_descs.Schema();
    ASSERT_EQ(schema.Size(), 3u);
    EXPECT_EQ(schema.Definition(0).name, "sterimol_L");
    EXPECT_EQ(schema.Definition(1).name, "sterimol_B1");
    EXPECT_EQ(schema.Definition(2).name, "sterimol_B5");

    // Verify that the c0-c6 bond (GetIdx 0 -> 6) is present with correct values
    // Expected from kallisto: C0(0)->C6(6): L=5.7049095404, B1=3.4638196900, B5=5.7437998791
    const auto& endpoints = bond_descs.BondEndpoints();
    std::optional<std::size_t> c0_c6_row;
    for (std::size_t i = 0; i < bond_descs.BondCount(); ++i) {
        const auto [origin, partner] = endpoints[i];
        if (origin == c0->GetIdx() && partner == c6->GetIdx()) {
            c0_c6_row = i;
            break;
        }
    }
    ASSERT_TRUE(c0_c6_row.has_value()) << "c0->c6 bond not found";

    auto l_val = bond_descs.Value(*c0_c6_row, 0);
    auto b1_val = bond_descs.Value(*c0_c6_row, 1);
    auto b5_val = bond_descs.Value(*c0_c6_row, 2);

    ASSERT_TRUE(l_val.has_value());
    ASSERT_TRUE(b1_val.has_value());
    ASSERT_TRUE(b5_val.has_value());

    EXPECT_NEAR(l_val.value(), 5.7049095404, 1e-5);
    EXPECT_NEAR(b1_val.value(), 3.4638196900, 1e-5);
    EXPECT_NEAR(b5_val.value(), 5.7437998791, 1e-5);

    // Also verify reverse direction c6->c0
    // Expected: C6(6)->C0(0): L=11.0994690152, B1=3.4638196900, B5=5.7437998791
    std::optional<std::size_t> c6_c0_row;
    for (std::size_t i = 0; i < bond_descs.BondCount(); ++i) {
        const auto [origin, partner] = endpoints[i];
        if (origin == c6->GetIdx() && partner == c0->GetIdx()) {
            c6_c0_row = i;
            break;
        }
    }
    ASSERT_TRUE(c6_c0_row.has_value()) << "c6->c0 bond not found";

    auto l_val2 = bond_descs.Value(*c6_c0_row, 0);
    auto b1_val2 = bond_descs.Value(*c6_c0_row, 1);
    auto b5_val2 = bond_descs.Value(*c6_c0_row, 2);

    ASSERT_TRUE(l_val2.has_value());
    ASSERT_TRUE(b1_val2.has_value());
    ASSERT_TRUE(b5_val2.has_value());

    EXPECT_NEAR(l_val2.value(), 11.0994690152, 1e-5);
    EXPECT_NEAR(b1_val2.value(), 3.4638196900, 1e-5);
    EXPECT_NEAR(b5_val2.value(), 5.7437998791, 1e-5);

    // Verify consistency: for each bond row, KallistoSterimol should give the same values
    for (std::size_t i = 0; i < bond_descs.BondCount(); ++i) {
        const auto [origin, partner] = endpoints[i];
        auto s = KallistoSterimol(mol, origin, partner);
        ASSERT_TRUE(s.has_value()) << "KallistoSterimol failed for bond " << origin << "->" << partner;

        auto l_table = bond_descs.Value(i, 0);
        auto b1_table = bond_descs.Value(i, 1);
        auto b5_table = bond_descs.Value(i, 2);

        ASSERT_TRUE(l_table.has_value());
        ASSERT_TRUE(b1_table.has_value());
        ASSERT_TRUE(b5_table.has_value());

        EXPECT_NEAR(l_table.value(), s->l, 1e-12) << "L mismatch for bond " << origin << "->" << partner;
        EXPECT_NEAR(b1_table.value(), s->b1, 1e-12) << "B1 mismatch for bond " << origin << "->" << partner;
        EXPECT_NEAR(b5_table.value(), s->b5, 1e-12) << "B5 mismatch for bond " << origin << "->" << partner;
    }
}

// Test MakeKallistoBondDescriptors with NON-CONTIGUOUS atom GetIdx
// Regression test for the copy-renumbering bug: OEGraphMol copy constructor
// renumbers atom IDs to 0..N-1, even if the original has non-contiguous IDs
// after DeleteAtom. This test verifies that MakeKallistoBondDescriptors:
// (1) does not crash/throw when original has non-contiguous IDs,
// (2) emits bond row_ids using the ORIGINAL non-contiguous GetIdx,
// (3) computes Sterimol values matching KallistoSterimol for the same bonds.
TEST(KallistoDescriptors, BondDescriptorsNonContiguousAtomIDs) {
    // Build a 5-atom chain with 3D coords, then delete the middle atom (idx 2)
    // to create non-contiguous atom IDs: original {0,1,2,3,4} -> remaining {0,1,3,4}.
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c0 = mol.NewAtom(6);  // Carbon
    OEChem::OEAtomBase* c1 = mol.NewAtom(6);
    OEChem::OEAtomBase* c2 = mol.NewAtom(6);  // Will be deleted
    OEChem::OEAtomBase* c3 = mol.NewAtom(6);
    OEChem::OEAtomBase* c4 = mol.NewAtom(6);

    mol.NewBond(c0, c1, 1);
    mol.NewBond(c1, c2, 1);
    mol.NewBond(c2, c3, 1);
    mol.NewBond(c3, c4, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_0[3] = {0.0, 0.0, 0.0};
    const double coords_1[3] = {2.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_2[3] = {5.6 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_3[3] = {8.4 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_4[3] = {11.2 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(c0, coords_0);
    mol.SetCoords(c1, coords_1);
    mol.SetCoords(c2, coords_2);
    mol.SetCoords(c3, coords_3);
    mol.SetCoords(c4, coords_4);
    mol.SetDimension(3);

    // Verify initial contiguous indices
    EXPECT_EQ(c0->GetIdx(), 0u);
    EXPECT_EQ(c1->GetIdx(), 1u);
    EXPECT_EQ(c2->GetIdx(), 2u);
    EXPECT_EQ(c3->GetIdx(), 3u);
    EXPECT_EQ(c4->GetIdx(), 4u);

    // Delete the middle atom (c2, idx 2)
    mol.DeleteAtom(c2);

    // Collect remaining atom indices via iteration (OpenEye order)
    std::vector<unsigned int> remaining_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        remaining_indices.push_back(atom->GetIdx());
    }

    // Verify non-contiguous: {0,1,3,4} NOT {0,1,2,3}
    ASSERT_EQ(remaining_indices.size(), 4u);
    EXPECT_EQ(remaining_indices[0], 0u);
    EXPECT_EQ(remaining_indices[1], 1u);
    EXPECT_EQ(remaining_indices[2], 3u);  // Not 2!
    EXPECT_EQ(remaining_indices[3], 4u);  // Not 3!

    // Compute bond descriptors (should NOT throw/crash)
    auto bond_descs = MakeKallistoBondDescriptors(mol);

    // After deleting c2, remaining bonds: c0-c1, c3-c4 (both acyclic, non-ring).
    // Expected: 4 directed bonds (both directions): c0->c1, c1->c0, c3->c4, c4->c3
    ASSERT_EQ(bond_descs.BondCount(), 4u);

    const auto& endpoints = bond_descs.BondEndpoints();

    // Verify that the row_ids use the ORIGINAL non-contiguous GetIdx (not renumbered 0..3)
    // Expected pairs (in any order):
    // (0,1), (1,0) for c0-c1 bond
    // (3,4), (4,3) for c3-c4 bond
    // NO pair should contain idx 2 (deleted), and row_ids should NOT be (0,1), (1,0), (2,3), (3,2)
    // (which would occur if using renumbered copy IDs).

    std::set<std::pair<std::uint32_t, std::uint32_t>> emitted_pairs;
    for (std::size_t i = 0; i < bond_descs.BondCount(); ++i) {
        emitted_pairs.insert(endpoints[i]);
    }

    // Verify expected pairs are present
    EXPECT_TRUE(emitted_pairs.count({0, 1}) == 1) << "c0->c1 bond missing";
    EXPECT_TRUE(emitted_pairs.count({1, 0}) == 1) << "c1->c0 bond missing";
    EXPECT_TRUE(emitted_pairs.count({3, 4}) == 1) << "c3->c4 bond missing";
    EXPECT_TRUE(emitted_pairs.count({4, 3}) == 1) << "c4->c3 bond missing";

    // Verify NO pair contains the deleted idx 2
    for (const auto& [origin, partner] : endpoints) {
        EXPECT_NE(origin, 2u) << "Row_id origin contains deleted idx 2";
        EXPECT_NE(partner, 2u) << "Row_id partner contains deleted idx 2";
    }

    // Verify Sterimol values match KallistoSterimol for the same POSITIONAL bonds.
    // For each bond row_id (origin, partner), the Sterimol values should match
    // KallistoSterimol(mol, origin, partner) using the ORIGINAL GetIdx.
    for (std::size_t i = 0; i < bond_descs.BondCount(); ++i) {
        const auto [origin, partner] = endpoints[i];

        // KallistoSterimol takes POSITIONAL indices (context array position).
        // The context array was built by iterating the original mol.GetAtoms()
        // in order, so the POSITIONAL index for GetIdx X is the position of X
        // in the remaining atoms after deletion.
        // For our molecule: GetIdx 0 -> position 0, GetIdx 1 -> position 1,
        //                   GetIdx 3 -> position 2, GetIdx 4 -> position 3.
        std::unordered_map<std::uint32_t, std::size_t> idx_to_position;
        std::size_t pos = 0;
        for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
            idx_to_position[atom->GetIdx()] = pos++;
        }

        const std::size_t pos_origin = idx_to_position.at(origin);
        const std::size_t pos_partner = idx_to_position.at(partner);

        auto s = KallistoSterimol(mol, pos_origin, pos_partner);
        ASSERT_TRUE(s.has_value()) << "KallistoSterimol failed for bond " << origin << "->" << partner;

        auto l_table = bond_descs.Value(i, 0);
        auto b1_table = bond_descs.Value(i, 1);
        auto b5_table = bond_descs.Value(i, 2);

        ASSERT_TRUE(l_table.has_value());
        ASSERT_TRUE(b1_table.has_value());
        ASSERT_TRUE(b5_table.has_value());

        EXPECT_NEAR(l_table.value(), s->l, 1e-12) << "L mismatch for bond " << origin << "->" << partner;
        EXPECT_NEAR(b1_table.value(), s->b1, 1e-12) << "B1 mismatch for bond " << origin << "->" << partner;
        EXPECT_NEAR(b5_table.value(), s->b5, 1e-12) << "B5 mismatch for bond " << origin << "->" << partner;
    }
}

// Test MakeKallistoAtomDescriptors on eligible 3D molecule fills all 8 columns
TEST(KallistoDescriptors, MakeKallistoAtomDescriptorsEligible) {
    // H2O: O at (0,0,0), H at (1.8,0,0), H at (0,1.8,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    auto result = MakeKallistoAtomDescriptors(mol);

    // Verify AtomCount > 0 and schema size = 8
    EXPECT_EQ(result.AtomCount(), 3u);
    EXPECT_EQ(result.Schema().Size(), 8u);

    // Verify all columns have values for all atoms
    for (std::size_t atom = 0; atom < result.AtomCount(); ++atom) {
        for (std::size_t col = 0; col < 8; ++col) {
            auto val = result.Value(atom, col);
            EXPECT_TRUE(val.has_value()) << "Missing value at atom " << atom << " col " << col;
        }
    }
}

// Test MakeKallistoAtomDescriptors on 2D molecule returns Empty()
TEST(KallistoDescriptors, MakeKallistoAtomDescriptors2D) {
    // 2D molecule
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c1 = mol.NewAtom(6);
    OEChem::OEAtomBase* c2 = mol.NewAtom(6);
    mol.NewBond(c1, c2, 1);

    const double coords_c1[3] = {0.0, 0.0, 0.0};
    const double coords_c2[3] = {1.5, 0.0, 0.0};
    mol.SetCoords(c1, coords_c1);
    mol.SetCoords(c2, coords_c2);
    mol.SetDimension(2);

    auto result = MakeKallistoAtomDescriptors(mol);
    EXPECT_EQ(result.AtomCount(), 0u);
}

// Test MakeKallistoAtomDescriptors on Z>86 molecule returns Empty()
TEST(KallistoDescriptors, MakeKallistoAtomDescriptorsHeavyElement) {
    // Francium, Z=87 > 86
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* fr = mol.NewAtom(87);
    mol.SetDimension(3);

    constexpr double coords[3] = {0.0, 0.0, 0.0};
    mol.SetCoords(fr, coords);

    auto result = MakeKallistoAtomDescriptors(mol);
    EXPECT_EQ(result.AtomCount(), 0u);
}

// Test KallistoAtomDescriptorSource::CalculateBatch over [valid, skipped(2D), valid]
// Verify: Size()==3, SegmentAtomCount(1)==0 (empty middle), total AtomCount == sum of valid,
// segment values match single-molecule Compute
TEST(KallistoDescriptors, AtomDescriptorSourceCalculateBatch) {
    // Build three molecules: H2O (valid), 2D C2 (skipped), H2O (valid)
    OEChem::OEGraphMol mol1;
    OEChem::OEAtomBase* o1 = mol1.NewAtom(8);
    OEChem::OEAtomBase* h1_1 = mol1.NewAtom(1);
    OEChem::OEAtomBase* h1_2 = mol1.NewAtom(1);
    mol1.NewBond(o1, h1_1, 1);
    mol1.NewBond(o1, h1_2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol1.SetCoords(o1, coords_o);
    mol1.SetCoords(h1_1, coords_h1);
    mol1.SetCoords(h1_2, coords_h2);
    mol1.SetDimension(3);

    // 2D molecule (skipped)
    OEChem::OEGraphMol mol2;
    OEChem::OEAtomBase* c2_1 = mol2.NewAtom(6);
    OEChem::OEAtomBase* c2_2 = mol2.NewAtom(6);
    mol2.NewBond(c2_1, c2_2, 1);
    const double coords_2d_c1[3] = {0.0, 0.0, 0.0};
    const double coords_2d_c2[3] = {1.5, 0.0, 0.0};
    mol2.SetCoords(c2_1, coords_2d_c1);
    mol2.SetCoords(c2_2, coords_2d_c2);
    mol2.SetDimension(2);

    // Another H2O (valid)
    OEChem::OEGraphMol mol3;
    OEChem::OEAtomBase* o3 = mol3.NewAtom(8);
    OEChem::OEAtomBase* h3_1 = mol3.NewAtom(1);
    OEChem::OEAtomBase* h3_2 = mol3.NewAtom(1);
    mol3.NewBond(o3, h3_1, 1);
    mol3.NewBond(o3, h3_2, 1);
    mol3.SetCoords(o3, coords_o);
    mol3.SetCoords(h3_1, coords_h1);
    mol3.SetCoords(h3_2, coords_h2);
    mol3.SetDimension(3);

    // Build pointer vector (create base references for proper type conversion)
    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    const OEChem::OEMolBase& base3 = mol3;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2, &base3};

    KallistoAtomDescriptorSource source;
    auto batch = source.CalculateBatch(mols);

    // Verify batch shape
    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.SegmentAtomCount(0), 3u);  // mol1 has 3 atoms
    EXPECT_EQ(batch.SegmentAtomCount(1), 0u);  // mol2 is skipped (empty segment)
    EXPECT_EQ(batch.SegmentAtomCount(2), 3u);  // mol3 has 3 atoms
    EXPECT_EQ(batch.AtomCount(), 6u);          // total = 3 + 0 + 3

    // Verify segment values match single-molecule Compute
    auto set1 = source.Compute(mol1);
    auto set3 = source.Compute(mol3);

    ASSERT_EQ(set1.AtomCount(), 3u);
    ASSERT_EQ(set3.AtomCount(), 3u);

    // Compare values for first segment (atoms 0-2 in batch)
    for (std::size_t atom = 0; atom < 3; ++atom) {
        for (std::size_t col = 0; col < 8; ++col) {
            auto batch_val = batch.ColumnDataAddress(col);
            auto batch_valid = batch.ColumnValidityAddress(col);
            auto set_val = set1.Value(atom, col);

            // Access batch value at position atom (first segment starts at offset 0)
            const double* batch_vals = reinterpret_cast<const double*>(batch_val);
            const std::uint8_t* batch_valids = reinterpret_cast<const std::uint8_t*>(batch_valid);

            if (set_val.has_value()) {
                EXPECT_TRUE(batch_valids[atom] != 0) << "Validity mismatch at atom " << atom << " col " << col;
                EXPECT_NEAR(batch_vals[atom], set_val.value(), 1e-12) << "Value mismatch at atom " << atom << " col " << col;
            } else {
                EXPECT_TRUE(batch_valids[atom] == 0) << "Validity mismatch at atom " << atom << " col " << col;
            }
        }
    }

    // Compare values for third segment (atoms 3-5 in batch, offset by 3)
    for (std::size_t atom = 0; atom < 3; ++atom) {
        for (std::size_t col = 0; col < 8; ++col) {
            auto batch_val = batch.ColumnDataAddress(col);
            auto batch_valid = batch.ColumnValidityAddress(col);
            auto set_val = set3.Value(atom, col);

            const double* batch_vals = reinterpret_cast<const double*>(batch_val);
            const std::uint8_t* batch_valids = reinterpret_cast<const std::uint8_t*>(batch_valid);

            if (set_val.has_value()) {
                EXPECT_TRUE(batch_valids[atom + 3] != 0) << "Validity mismatch at atom " << atom + 3 << " col " << col;
                EXPECT_NEAR(batch_vals[atom + 3], set_val.value(), 1e-12) << "Value mismatch at atom " << atom + 3 << " col " << col;
            } else {
                EXPECT_TRUE(batch_valids[atom + 3] == 0) << "Validity mismatch at atom " << atom + 3 << " col " << col;
            }
        }
    }
}

// Test KallistoAtomDescriptorSource::CalculateBatch throws on null pointer
TEST(KallistoDescriptors, AtomDescriptorSourceCalculateBatchNullPointer) {
    OEChem::OEGraphMol mol1;
    OEChem::OEAtomBase* o1 = mol1.NewAtom(8);
    mol1.SetDimension(3);
    const double coords[3] = {0.0, 0.0, 0.0};
    mol1.SetCoords(o1, coords);

    const OEChem::OEMolBase& base1 = mol1;
    std::vector<const OEChem::OEMolBase*> mols{&base1, nullptr};

    KallistoAtomDescriptorSource source;
    EXPECT_THROW(source.CalculateBatch(mols), std::invalid_argument);
}

// Test KallistoBondDescriptorSource::CalculateBatch over [valid, skipped(2D), valid]
TEST(KallistoDescriptors, BondDescriptorSourceCalculateBatch) {
    // Build three molecules: simple acyclic chain (valid), 2D (skipped), another chain (valid)

    // Mol1: 3-atom acyclic chain C-C-C with 3D coords
    OEChem::OEGraphMol mol1;
    OEChem::OEAtomBase* c1_0 = mol1.NewAtom(6);
    OEChem::OEAtomBase* c1_1 = mol1.NewAtom(6);
    OEChem::OEAtomBase* c1_2 = mol1.NewAtom(6);
    mol1.NewBond(c1_0, c1_1, 1);
    mol1.NewBond(c1_1, c1_2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_0[3] = {0.0, 0.0, 0.0};
    const double coords_1[3] = {2.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_2[3] = {5.6 * bohr_to_angstrom, 0.0, 0.0};
    mol1.SetCoords(c1_0, coords_0);
    mol1.SetCoords(c1_1, coords_1);
    mol1.SetCoords(c1_2, coords_2);
    mol1.SetDimension(3);

    // Mol2: 2D (skipped)
    OEChem::OEGraphMol mol2;
    OEChem::OEAtomBase* c2_1 = mol2.NewAtom(6);
    OEChem::OEAtomBase* c2_2 = mol2.NewAtom(6);
    mol2.NewBond(c2_1, c2_2, 1);
    const double coords_2d_c1[3] = {0.0, 0.0, 0.0};
    const double coords_2d_c2[3] = {1.5, 0.0, 0.0};
    mol2.SetCoords(c2_1, coords_2d_c1);
    mol2.SetCoords(c2_2, coords_2d_c2);
    mol2.SetDimension(2);

    // Mol3: another 3-atom acyclic chain
    OEChem::OEGraphMol mol3;
    OEChem::OEAtomBase* c3_0 = mol3.NewAtom(6);
    OEChem::OEAtomBase* c3_1 = mol3.NewAtom(6);
    OEChem::OEAtomBase* c3_2 = mol3.NewAtom(6);
    mol3.NewBond(c3_0, c3_1, 1);
    mol3.NewBond(c3_1, c3_2, 1);
    mol3.SetCoords(c3_0, coords_0);
    mol3.SetCoords(c3_1, coords_1);
    mol3.SetCoords(c3_2, coords_2);
    mol3.SetDimension(3);

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    const OEChem::OEMolBase& base3 = mol3;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2, &base3};

    KallistoBondDescriptorSource source;
    auto batch = source.CalculateBatch(mols);

    // Verify batch shape
    // mol1 has 2 acyclic bonds (c0-c1, c1-c2) × 2 directions = 4 bond rows
    // mol2 is skipped → 0 bond rows
    // mol3 has 2 acyclic bonds × 2 directions = 4 bond rows
    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.SegmentBondCount(0), 4u);
    EXPECT_EQ(batch.SegmentBondCount(1), 0u);  // empty middle segment
    EXPECT_EQ(batch.SegmentBondCount(2), 4u);
    EXPECT_EQ(batch.BondCount(), 8u);  // total = 4 + 0 + 4

    // Verify segment values match single-molecule Compute
    auto set1 = source.Compute(mol1);
    auto set3 = source.Compute(mol3);

    ASSERT_EQ(set1.BondCount(), 4u);
    ASSERT_EQ(set3.BondCount(), 4u);

    // Compare values for first segment (bonds 0-3 in batch)
    for (std::size_t bond = 0; bond < 4; ++bond) {
        for (std::size_t col = 0; col < 3; ++col) {
            auto batch_val = batch.ColumnDataAddress(col);
            auto batch_valid = batch.ColumnValidityAddress(col);
            auto set_val = set1.Value(bond, col);

            const double* batch_vals = reinterpret_cast<const double*>(batch_val);
            const std::uint8_t* batch_valids = reinterpret_cast<const std::uint8_t*>(batch_valid);

            if (set_val.has_value()) {
                EXPECT_TRUE(batch_valids[bond] != 0) << "Validity mismatch at bond " << bond << " col " << col;
                EXPECT_NEAR(batch_vals[bond], set_val.value(), 1e-12) << "Value mismatch at bond " << bond << " col " << col;
            } else {
                EXPECT_TRUE(batch_valids[bond] == 0) << "Validity mismatch at bond " << bond << " col " << col;
            }
        }
    }

    // Compare values for third segment (bonds 4-7 in batch, offset by 4)
    for (std::size_t bond = 0; bond < 4; ++bond) {
        for (std::size_t col = 0; col < 3; ++col) {
            auto batch_val = batch.ColumnDataAddress(col);
            auto batch_valid = batch.ColumnValidityAddress(col);
            auto set_val = set3.Value(bond, col);

            const double* batch_vals = reinterpret_cast<const double*>(batch_val);
            const std::uint8_t* batch_valids = reinterpret_cast<const std::uint8_t*>(batch_valid);

            if (set_val.has_value()) {
                EXPECT_TRUE(batch_valids[bond + 4] != 0) << "Validity mismatch at bond " << bond + 4 << " col " << col;
                EXPECT_NEAR(batch_vals[bond + 4], set_val.value(), 1e-12) << "Value mismatch at bond " << bond + 4 << " col " << col;
            } else {
                EXPECT_TRUE(batch_valids[bond + 4] == 0) << "Validity mismatch at bond " << bond + 4 << " col " << col;
            }
        }
    }
}

// Test KallistoBondDescriptorSource::CalculateBatch throws on null pointer
TEST(KallistoDescriptors, BondDescriptorSourceCalculateBatchNullPointer) {
    OEChem::OEGraphMol mol1;
    OEChem::OEAtomBase* c1 = mol1.NewAtom(6);
    OEChem::OEAtomBase* c2 = mol1.NewAtom(6);
    mol1.NewBond(c1, c2, 1);
    mol1.SetDimension(3);
    const double coords[3] = {0.0, 0.0, 0.0};
    mol1.SetCoords(c1, coords);
    mol1.SetCoords(c2, coords);

    const OEChem::OEMolBase& base1 = mol1;
    std::vector<const OEChem::OEMolBase*> mols{&base1, nullptr};

    KallistoBondDescriptorSource source;
    EXPECT_THROW(source.CalculateBatch(mols), std::invalid_argument);
}

}  // namespace
}  // namespace OEFP
