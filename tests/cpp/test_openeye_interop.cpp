#include <gtest/gtest.h>

#include "oefp/compare.h"
#include "oefp/metric.h"
#include "oefp/openeye.h"

#include <oechem.h>
#include <oegraphsim.h>

#include <stdexcept>

namespace OEFP {
namespace test {

TEST(OpenEyeInteropTest, ImportsGeneratedFingerprintAndMatchesOpenEyeTanimoto) {
    OEChem::OEGraphMol mol_a;
    OEChem::OEGraphMol mol_b;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol_a, "c1ccccc1"));
    ASSERT_TRUE(OEChem::OESmilesToMol(mol_b, "c1ccc(O)cc1"));

    OEGraphSim::OEFingerPrint oe_a;
    OEGraphSim::OEFingerPrint oe_b;
    ASSERT_TRUE(OEGraphSim::OEMakeCircularFP(oe_a, mol_a));
    ASSERT_TRUE(OEGraphSim::OEMakeCircularFP(oe_b, mol_b));

    const auto fp_a = FromOEFingerPrint(oe_a);
    const auto fp_b = FromOEFingerPrint(oe_b);

    EXPECT_EQ(fp_a.SizeBits(), oe_a.GetSize());
    EXPECT_EQ(fp_b.SizeBits(), oe_b.GetSize());
    EXPECT_NEAR(Compare(fp_a, fp_b, Metric::Tanimoto()), OEGraphSim::OETanimoto(oe_a, oe_b), 1.0e-7);
}

TEST(OpenEyeInteropTest, ExportRoundTripsGeneratedFingerprintBits) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));

    OEGraphSim::OEFingerPrint original;
    ASSERT_TRUE(OEGraphSim::OEMakeCircularFP(original, mol));

    const auto imported = FromOEFingerPrint(original);
    const auto exported = ToOEFingerPrint(imported);

    EXPECT_EQ(exported.GetSize(), original.GetSize());
    EXPECT_EQ(exported.CountBits(), original.CountBits());
    EXPECT_TRUE(OEGraphSim::OEIsSameFPType(original, exported));
    EXPECT_NEAR(OEGraphSim::OETanimoto(original, exported), 1.0, 1.0e-7);
}

TEST(OpenEyeInteropTest, ExportRejectsNonOpenEyeFingerprintSpec) {
    FingerprintSpec spec;
    spec.size_bits = 64;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "unit-test";
    spec.source_type = "manual";

    EXPECT_THROW(ToOEFingerPrint(OEFP(spec)), std::invalid_argument);
}

} // namespace test
} // namespace OEFP
