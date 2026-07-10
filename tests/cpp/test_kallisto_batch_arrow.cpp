#include "oefp/atom_descriptor_arrow.h"
#include "oefp/atom_descriptor.h"
#include "oefp/kallisto_descriptors.h"

#include <arrow/api.h>

#include <gtest/gtest.h>
#include <oechem.h>

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace OEFP {
namespace {

// Helper: build a 3D ethane molecule
OEChem::OEGraphMol make_ethane() {
    OEChem::OEGraphMol mol;
    auto* c1 = mol.NewAtom(6);  // Carbon
    auto* c2 = mol.NewAtom(6);
    auto* h1 = mol.NewAtom(1);  // Hydrogen
    auto* h2 = mol.NewAtom(1);
    auto* h3 = mol.NewAtom(1);
    auto* h4 = mol.NewAtom(1);
    auto* h5 = mol.NewAtom(1);
    auto* h6 = mol.NewAtom(1);

    mol.NewBond(c1, c2, 1);
    mol.NewBond(c1, h1, 1);
    mol.NewBond(c1, h2, 1);
    mol.NewBond(c1, h3, 1);
    mol.NewBond(c2, h4, 1);
    mol.NewBond(c2, h5, 1);
    mol.NewBond(c2, h6, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double cc_dist = 1.54 / bohr_to_angstrom;  // C-C in Bohr
    const double ch_dist = 1.09 / bohr_to_angstrom;  // C-H in Bohr

    const double coords_c1[3] = {-cc_dist * 0.5, 0.0, 0.0};
    const double coords_c2[3] = {cc_dist * 0.5, 0.0, 0.0};
    const double coords_h1[3] = {-cc_dist * 0.5 - ch_dist, 0.0, 0.0};
    const double coords_h2[3] = {-cc_dist * 0.5, ch_dist * 0.866, ch_dist * 0.5};
    const double coords_h3[3] = {-cc_dist * 0.5, -ch_dist * 0.866, ch_dist * 0.5};
    const double coords_h4[3] = {cc_dist * 0.5 + ch_dist, 0.0, 0.0};
    const double coords_h5[3] = {cc_dist * 0.5, ch_dist * 0.866, -ch_dist * 0.5};
    const double coords_h6[3] = {cc_dist * 0.5, -ch_dist * 0.866, -ch_dist * 0.5};

    mol.SetCoords(c1, coords_c1);
    mol.SetCoords(c2, coords_c2);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetCoords(h3, coords_h3);
    mol.SetCoords(h4, coords_h4);
    mol.SetCoords(h5, coords_h5);
    mol.SetCoords(h6, coords_h6);

    mol.SetDimension(3);
    return mol;
}

// Helper: build a 2D molecule (ineligible)
OEChem::OEGraphMol make_2d_molecule() {
    OEChem::OEGraphMol mol;
    mol.NewAtom(6);  // Carbon
    mol.SetDimension(2);
    return mol;
}

// Helper: build a molecule with Z > 86 (ineligible)
OEChem::OEGraphMol make_heavy_molecule() {
    OEChem::OEGraphMol mol;
    auto* atom = mol.NewAtom(92);  // Uranium (Z=92 > 86)
    const double coords[3] = {0.0, 0.0, 0.0};
    mol.SetCoords(atom, coords);
    mol.SetDimension(3);
    return mol;
}

TEST(KallistoBatchArrow, AtomDescriptorRoundTripWithEmptySegments) {
    // Build batch: [valid, skipped (2D), valid]
    auto mol1 = make_ethane();
    auto mol2 = make_2d_molecule();
    auto mol3 = make_ethane();

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    const OEChem::OEMolBase& base3 = mol3;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2, &base3};

    KallistoAtomDescriptorSource source;
    const auto original_batch = source.CalculateBatch(mols);

    ASSERT_EQ(original_batch.Size(), 3u);
    EXPECT_GT(original_batch.SegmentAtomCount(0), 0u);
    EXPECT_EQ(original_batch.SegmentAtomCount(1), 0u);  // Empty (skipped 2D)
    EXPECT_GT(original_batch.SegmentAtomCount(2), 0u);

    const auto original_atom_count = original_batch.AtomCount();
    EXPECT_GT(original_atom_count, 0u);

    // Convert to Arrow and back
    const auto arrow_rb = AtomDescriptorBatchToArrow(original_batch);
    ASSERT_NE(arrow_rb, nullptr);

    const auto reconstructed_batch = AtomDescriptorBatchFromArrow(arrow_rb);

    // Verify round-trip
    ASSERT_EQ(reconstructed_batch.Size(), original_batch.Size());
    EXPECT_EQ(reconstructed_batch.AtomCount(), original_atom_count);

    for (std::size_t mol = 0; mol < original_batch.Size(); ++mol) {
        EXPECT_EQ(reconstructed_batch.SegmentAtomCount(mol),
                  original_batch.SegmentAtomCount(mol))
            << "Segment " << mol << " atom count mismatch";
    }

    // Verify atom indices
    const auto* orig_indices = reinterpret_cast<const std::uint32_t*>(
        original_batch.AtomIndexDataAddress());
    const auto* recon_indices = reinterpret_cast<const std::uint32_t*>(
        reconstructed_batch.AtomIndexDataAddress());

    for (std::size_t i = 0; i < original_atom_count; ++i) {
        EXPECT_EQ(recon_indices[i], orig_indices[i])
            << "Atom index " << i << " mismatch";
    }

    // Verify descriptor values and validity for each column
    const auto& schema = original_batch.Schema();
    for (std::size_t col = 0; col < schema.Size(); ++col) {
        const auto* orig_values = reinterpret_cast<const double*>(
            original_batch.ColumnDataAddress(col));
        const auto* recon_values = reinterpret_cast<const double*>(
            reconstructed_batch.ColumnDataAddress(col));
        const auto* orig_validity = reinterpret_cast<const std::uint8_t*>(
            original_batch.ColumnValidityAddress(col));
        const auto* recon_validity = reinterpret_cast<const std::uint8_t*>(
            reconstructed_batch.ColumnValidityAddress(col));

        for (std::size_t i = 0; i < original_atom_count; ++i) {
            EXPECT_EQ(recon_validity[i], orig_validity[i])
                << "Column " << col << " atom " << i << " validity mismatch";
            if (orig_validity[i] != 0u) {
                EXPECT_DOUBLE_EQ(recon_values[i], orig_values[i])
                    << "Column " << col << " atom " << i << " value mismatch";
            }
        }
    }

    // Verify schema identity is preserved (native schema singleton)
    EXPECT_EQ(reconstructed_batch.Schema().SchemaId(),
              KallistoAtomDescriptorSchema()->SchemaId())
        << "Round-tripped batch must preserve native schema identity";

    // Verify append compatibility: a fresh native-schema batch can append the round-tripped data
    auto native_batch = AtomDescriptorBatch::Empty(KallistoAtomDescriptorSchema());
    std::size_t atom_offset = 0;
    for (std::size_t mol = 0; mol < reconstructed_batch.Size(); ++mol) {
        const auto seg_atom_count = reconstructed_batch.SegmentAtomCount(mol);
        if (seg_atom_count == 0) {
            EXPECT_NO_THROW(native_batch.Append(AtomDescriptorSet::Empty(KallistoAtomDescriptorSchema())))
                << "Appending empty segment " << mol << " must succeed";
            continue;
        }

        // Extract atom indices for this segment
        std::vector<std::uint32_t> atom_indices(seg_atom_count);
        const auto* indices_data = reinterpret_cast<const std::uint32_t*>(
            reconstructed_batch.AtomIndexDataAddress());
        for (std::size_t i = 0; i < seg_atom_count; ++i) {
            atom_indices[i] = indices_data[atom_offset + i];
        }

        // Extract column values for this segment
        std::vector<std::vector<std::optional<double>>> columns(schema.Size());
        for (std::size_t col = 0; col < schema.Size(); ++col) {
            columns[col].resize(seg_atom_count);
            const auto* values = reinterpret_cast<const double*>(
                reconstructed_batch.ColumnDataAddress(col));
            const auto* validity = reinterpret_cast<const std::uint8_t*>(
                reconstructed_batch.ColumnValidityAddress(col));
            for (std::size_t i = 0; i < seg_atom_count; ++i) {
                if (validity[atom_offset + i] != 0u) {
                    columns[col][i] = values[atom_offset + i];
                } else {
                    columns[col][i] = std::nullopt;
                }
            }
        }

        EXPECT_NO_THROW(native_batch.Append(AtomDescriptorSet(
            KallistoAtomDescriptorSchema(), std::move(atom_indices), std::move(columns))))
            << "Appending reconstructed segment " << mol << " to native-schema batch must succeed";

        atom_offset += seg_atom_count;
    }
}

TEST(KallistoBatchArrow, BondDescriptorRoundTripWithEmptySegments) {
    // Build batch: [valid, skipped (heavy), valid]
    auto mol1 = make_ethane();
    auto mol2 = make_heavy_molecule();
    auto mol3 = make_ethane();

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    const OEChem::OEMolBase& base3 = mol3;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2, &base3};

    KallistoBondDescriptorSource source;
    const auto original_batch = source.CalculateBatch(mols);

    ASSERT_EQ(original_batch.Size(), 3u);
    EXPECT_GT(original_batch.SegmentBondCount(0), 0u);
    EXPECT_EQ(original_batch.SegmentBondCount(1), 0u);  // Empty (skipped heavy)
    EXPECT_GT(original_batch.SegmentBondCount(2), 0u);

    const auto original_bond_count = original_batch.BondCount();
    EXPECT_GT(original_bond_count, 0u);

    // Convert to Arrow and back
    const auto arrow_rb = BondDescriptorBatchToArrow(original_batch);
    ASSERT_NE(arrow_rb, nullptr);

    const auto reconstructed_batch = BondDescriptorBatchFromArrow(arrow_rb);

    // Verify round-trip
    ASSERT_EQ(reconstructed_batch.Size(), original_batch.Size());
    EXPECT_EQ(reconstructed_batch.BondCount(), original_bond_count);

    for (std::size_t mol = 0; mol < original_batch.Size(); ++mol) {
        EXPECT_EQ(reconstructed_batch.SegmentBondCount(mol),
                  original_batch.SegmentBondCount(mol))
            << "Segment " << mol << " bond count mismatch";
    }

    // Verify bond endpoints
    const auto* orig_begin = reinterpret_cast<const std::uint32_t*>(
        original_batch.BondBeginDataAddress());
    const auto* orig_end = reinterpret_cast<const std::uint32_t*>(
        original_batch.BondEndDataAddress());
    const auto* recon_begin = reinterpret_cast<const std::uint32_t*>(
        reconstructed_batch.BondBeginDataAddress());
    const auto* recon_end = reinterpret_cast<const std::uint32_t*>(
        reconstructed_batch.BondEndDataAddress());

    for (std::size_t i = 0; i < original_bond_count; ++i) {
        EXPECT_EQ(recon_begin[i], orig_begin[i])
            << "Bond " << i << " begin mismatch";
        EXPECT_EQ(recon_end[i], orig_end[i])
            << "Bond " << i << " end mismatch";
    }

    // Verify descriptor values and validity for each column
    const auto& schema = original_batch.Schema();
    for (std::size_t col = 0; col < schema.Size(); ++col) {
        const auto* orig_values = reinterpret_cast<const double*>(
            original_batch.ColumnDataAddress(col));
        const auto* recon_values = reinterpret_cast<const double*>(
            reconstructed_batch.ColumnDataAddress(col));
        const auto* orig_validity = reinterpret_cast<const std::uint8_t*>(
            original_batch.ColumnValidityAddress(col));
        const auto* recon_validity = reinterpret_cast<const std::uint8_t*>(
            reconstructed_batch.ColumnValidityAddress(col));

        for (std::size_t i = 0; i < original_bond_count; ++i) {
            EXPECT_EQ(recon_validity[i], orig_validity[i])
                << "Column " << col << " bond " << i << " validity mismatch";
            if (orig_validity[i] != 0u) {
                EXPECT_DOUBLE_EQ(recon_values[i], orig_values[i])
                    << "Column " << col << " bond " << i << " value mismatch";
            }
        }
    }

    // Verify schema identity is preserved (native schema singleton)
    EXPECT_EQ(reconstructed_batch.Schema().SchemaId(),
              KallistoBondDescriptorSchema()->SchemaId())
        << "Round-tripped batch must preserve native schema identity";

    // Verify append compatibility: a fresh native-schema batch can append the round-tripped data
    auto native_batch = BondDescriptorBatch::Empty(KallistoBondDescriptorSchema());
    std::size_t bond_offset = 0;
    for (std::size_t mol = 0; mol < reconstructed_batch.Size(); ++mol) {
        const auto seg_bond_count = reconstructed_batch.SegmentBondCount(mol);
        if (seg_bond_count == 0) {
            EXPECT_NO_THROW(native_batch.Append(BondDescriptorSet::Empty(KallistoBondDescriptorSchema())))
                << "Appending empty segment " << mol << " must succeed";
            continue;
        }

        // Extract bond endpoints for this segment
        std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints(seg_bond_count);
        const auto* begin_data = reinterpret_cast<const std::uint32_t*>(
            reconstructed_batch.BondBeginDataAddress());
        const auto* end_data = reinterpret_cast<const std::uint32_t*>(
            reconstructed_batch.BondEndDataAddress());
        for (std::size_t i = 0; i < seg_bond_count; ++i) {
            bond_endpoints[i] = {begin_data[bond_offset + i], end_data[bond_offset + i]};
        }

        // Extract column values for this segment
        std::vector<std::vector<std::optional<double>>> columns(schema.Size());
        for (std::size_t col = 0; col < schema.Size(); ++col) {
            columns[col].resize(seg_bond_count);
            const auto* values = reinterpret_cast<const double*>(
                reconstructed_batch.ColumnDataAddress(col));
            const auto* validity = reinterpret_cast<const std::uint8_t*>(
                reconstructed_batch.ColumnValidityAddress(col));
            for (std::size_t i = 0; i < seg_bond_count; ++i) {
                if (validity[bond_offset + i] != 0u) {
                    columns[col][i] = values[bond_offset + i];
                } else {
                    columns[col][i] = std::nullopt;
                }
            }
        }

        EXPECT_NO_THROW(native_batch.Append(BondDescriptorSet(
            KallistoBondDescriptorSchema(), std::move(bond_endpoints), std::move(columns))))
            << "Appending reconstructed segment " << mol << " to native-schema batch must succeed";

        bond_offset += seg_bond_count;
    }
}

TEST(KallistoBatchArrow, AtomDescriptorIpcRoundTrip) {
    auto mol1 = make_ethane();
    auto mol2 = make_2d_molecule();

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2};

    KallistoAtomDescriptorSource source;
    const auto original_batch = source.CalculateBatch(mols);

    const std::string path = "/tmp/kallisto_atom_test.arrow";
    WriteKallistoAtomIpc(original_batch, path);

    const auto reconstructed_batch = ReadKallistoAtomIpc(path);

    ASSERT_EQ(reconstructed_batch.Size(), original_batch.Size());
    EXPECT_EQ(reconstructed_batch.AtomCount(), original_batch.AtomCount());
}

TEST(KallistoBatchArrow, BondDescriptorIpcRoundTrip) {
    auto mol1 = make_ethane();
    auto mol2 = make_heavy_molecule();

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2};

    KallistoBondDescriptorSource source;
    const auto original_batch = source.CalculateBatch(mols);

    const std::string path = "/tmp/kallisto_bond_test.arrow";
    WriteKallistoBondIpc(original_batch, path);

    const auto reconstructed_batch = ReadKallistoBondIpc(path);

    ASSERT_EQ(reconstructed_batch.Size(), original_batch.Size());
    EXPECT_EQ(reconstructed_batch.BondCount(), original_batch.BondCount());
}

TEST(KallistoBatchArrow, AtomDescriptorRejectsSwappedColumns) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoAtomDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = AtomDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_columns(), 3);

    // Build a record batch with swapped feature columns (2 and 3)
    std::vector<std::shared_ptr<arrow::Field>> swapped_fields;
    std::vector<std::shared_ptr<arrow::Array>> swapped_arrays;

    for (int i = 0; i < valid_rb->num_columns(); ++i) {
        if (i == 2) {
            swapped_fields.push_back(valid_rb->schema()->field(3));
            swapped_arrays.push_back(valid_rb->column(3));
        } else if (i == 3) {
            swapped_fields.push_back(valid_rb->schema()->field(2));
            swapped_arrays.push_back(valid_rb->column(2));
        } else {
            swapped_fields.push_back(valid_rb->schema()->field(i));
            swapped_arrays.push_back(valid_rb->column(i));
        }
    }

    auto swapped_schema = arrow::schema(swapped_fields, valid_rb->schema()->metadata());
    auto swapped_rb = arrow::RecordBatch::Make(
        swapped_schema, valid_rb->num_rows(), swapped_arrays);

    EXPECT_THROW({
        AtomDescriptorBatchFromArrow(swapped_rb);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, AtomDescriptorRejectsRenamedColumn) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoAtomDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = AtomDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_columns(), 2);

    // Build a record batch with a renamed feature column
    std::vector<std::shared_ptr<arrow::Field>> renamed_fields;
    std::vector<std::shared_ptr<arrow::Array>> renamed_arrays;

    for (int i = 0; i < valid_rb->num_columns(); ++i) {
        if (i == 2) {
            renamed_fields.push_back(arrow::field("bogus_column", arrow::float64()));
        } else {
            renamed_fields.push_back(valid_rb->schema()->field(i));
        }
        renamed_arrays.push_back(valid_rb->column(i));
    }

    auto renamed_schema = arrow::schema(renamed_fields, valid_rb->schema()->metadata());
    auto renamed_rb = arrow::RecordBatch::Make(
        renamed_schema, valid_rb->num_rows(), renamed_arrays);

    EXPECT_THROW({
        AtomDescriptorBatchFromArrow(renamed_rb);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, AtomDescriptorRejectsWrongIdColumnType) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoAtomDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = AtomDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);

    // Build a record batch with molecule_id as int64 instead of uint32
    std::vector<std::shared_ptr<arrow::Field>> wrong_type_fields;
    std::vector<std::shared_ptr<arrow::Array>> wrong_type_arrays;

    for (int i = 0; i < valid_rb->num_columns(); ++i) {
        if (i == 0) {
            wrong_type_fields.push_back(arrow::field("molecule_id", arrow::int64()));
            // Convert uint32 array to int64
            arrow::Int64Builder builder;
            const auto uint32_array = std::dynamic_pointer_cast<arrow::UInt32Array>(valid_rb->column(0));
            for (std::int64_t j = 0; j < uint32_array->length(); ++j) {
                (void)builder.Append(static_cast<std::int64_t>(uint32_array->Value(j)));
            }
            std::shared_ptr<arrow::Array> int64_array;
            (void)builder.Finish(&int64_array);
            wrong_type_arrays.push_back(int64_array);
        } else {
            wrong_type_fields.push_back(valid_rb->schema()->field(i));
            wrong_type_arrays.push_back(valid_rb->column(i));
        }
    }

    auto wrong_type_schema = arrow::schema(wrong_type_fields, valid_rb->schema()->metadata());
    auto wrong_type_rb = arrow::RecordBatch::Make(
        wrong_type_schema, valid_rb->num_rows(), wrong_type_arrays);

    EXPECT_THROW({
        AtomDescriptorBatchFromArrow(wrong_type_rb);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, BondDescriptorRejectsSwappedColumns) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoBondDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = BondDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_columns(), 4);

    // Build a record batch with swapped feature columns (3 and 4)
    std::vector<std::shared_ptr<arrow::Field>> swapped_fields;
    std::vector<std::shared_ptr<arrow::Array>> swapped_arrays;

    for (int i = 0; i < valid_rb->num_columns(); ++i) {
        if (i == 3) {
            swapped_fields.push_back(valid_rb->schema()->field(4));
            swapped_arrays.push_back(valid_rb->column(4));
        } else if (i == 4) {
            swapped_fields.push_back(valid_rb->schema()->field(3));
            swapped_arrays.push_back(valid_rb->column(3));
        } else {
            swapped_fields.push_back(valid_rb->schema()->field(i));
            swapped_arrays.push_back(valid_rb->column(i));
        }
    }

    auto swapped_schema = arrow::schema(swapped_fields, valid_rb->schema()->metadata());
    auto swapped_rb = arrow::RecordBatch::Make(
        swapped_schema, valid_rb->num_rows(), swapped_arrays);

    EXPECT_THROW({
        BondDescriptorBatchFromArrow(swapped_rb);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, BondDescriptorRejectsRenamedColumn) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoBondDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = BondDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_columns(), 3);

    // Build a record batch with a renamed feature column
    std::vector<std::shared_ptr<arrow::Field>> renamed_fields;
    std::vector<std::shared_ptr<arrow::Array>> renamed_arrays;

    for (int i = 0; i < valid_rb->num_columns(); ++i) {
        if (i == 3) {
            renamed_fields.push_back(arrow::field("bogus_column", arrow::float64()));
        } else {
            renamed_fields.push_back(valid_rb->schema()->field(i));
        }
        renamed_arrays.push_back(valid_rb->column(i));
    }

    auto renamed_schema = arrow::schema(renamed_fields, valid_rb->schema()->metadata());
    auto renamed_rb = arrow::RecordBatch::Make(
        renamed_schema, valid_rb->num_rows(), renamed_arrays);

    EXPECT_THROW({
        BondDescriptorBatchFromArrow(renamed_rb);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, BondDescriptorRejectsWrongIdColumnName) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoBondDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = BondDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);

    // Build a record batch with begin renamed to start
    std::vector<std::shared_ptr<arrow::Field>> wrong_name_fields;
    std::vector<std::shared_ptr<arrow::Array>> wrong_name_arrays;

    for (int i = 0; i < valid_rb->num_columns(); ++i) {
        if (i == 1) {
            wrong_name_fields.push_back(arrow::field("start", arrow::uint32()));
        } else {
            wrong_name_fields.push_back(valid_rb->schema()->field(i));
        }
        wrong_name_arrays.push_back(valid_rb->column(i));
    }

    auto wrong_name_schema = arrow::schema(wrong_name_fields, valid_rb->schema()->metadata());
    auto wrong_name_rb = arrow::RecordBatch::Make(
        wrong_name_schema, valid_rb->num_rows(), wrong_name_arrays);

    EXPECT_THROW({
        BondDescriptorBatchFromArrow(wrong_name_rb);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, AtomDescriptorRejectsNullMoleculeId) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoAtomDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = AtomDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_rows(), 0);

    // Build a record batch with a null molecule_id in the first row
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    arrow::UInt32Builder molecule_id_builder;
    (void)molecule_id_builder.AppendNull();  // First row: null
    for (std::int64_t i = 1; i < valid_rb->num_rows(); ++i) {
        (void)molecule_id_builder.Append(
            std::dynamic_pointer_cast<arrow::UInt32Array>(valid_rb->column(0))->Value(i));
    }
    std::shared_ptr<arrow::Array> molecule_id_array;
    (void)molecule_id_builder.Finish(&molecule_id_array);

    fields.push_back(arrow::field("molecule_id", arrow::uint32()));  // nullable
    arrays.push_back(molecule_id_array);

    // Copy remaining columns
    for (int i = 1; i < valid_rb->num_columns(); ++i) {
        fields.push_back(valid_rb->schema()->field(i));
        arrays.push_back(valid_rb->column(i));
    }

    auto schema = arrow::schema(fields, valid_rb->schema()->metadata());
    auto rb_with_null = arrow::RecordBatch::Make(schema, valid_rb->num_rows(), arrays);

    EXPECT_THROW({
        AtomDescriptorBatchFromArrow(rb_with_null);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, AtomDescriptorRejectsNullAtomIndex) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoAtomDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = AtomDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_rows(), 0);

    // Build a record batch with a null atom_index in the first row
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    fields.push_back(valid_rb->schema()->field(0));
    arrays.push_back(valid_rb->column(0));

    arrow::UInt32Builder atom_index_builder;
    (void)atom_index_builder.AppendNull();  // First row: null
    for (std::int64_t i = 1; i < valid_rb->num_rows(); ++i) {
        (void)atom_index_builder.Append(
            std::dynamic_pointer_cast<arrow::UInt32Array>(valid_rb->column(1))->Value(i));
    }
    std::shared_ptr<arrow::Array> atom_index_array;
    (void)atom_index_builder.Finish(&atom_index_array);

    fields.push_back(arrow::field("atom_index", arrow::uint32()));  // nullable
    arrays.push_back(atom_index_array);

    // Copy remaining columns
    for (int i = 2; i < valid_rb->num_columns(); ++i) {
        fields.push_back(valid_rb->schema()->field(i));
        arrays.push_back(valid_rb->column(i));
    }

    auto schema = arrow::schema(fields, valid_rb->schema()->metadata());
    auto rb_with_null = arrow::RecordBatch::Make(schema, valid_rb->num_rows(), arrays);

    EXPECT_THROW({
        AtomDescriptorBatchFromArrow(rb_with_null);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, BondDescriptorRejectsNullMoleculeId) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoBondDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = BondDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_rows(), 0);

    // Build a record batch with a null molecule_id in the first row
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    arrow::UInt32Builder molecule_id_builder;
    (void)molecule_id_builder.AppendNull();  // First row: null
    for (std::int64_t i = 1; i < valid_rb->num_rows(); ++i) {
        (void)molecule_id_builder.Append(
            std::dynamic_pointer_cast<arrow::UInt32Array>(valid_rb->column(0))->Value(i));
    }
    std::shared_ptr<arrow::Array> molecule_id_array;
    (void)molecule_id_builder.Finish(&molecule_id_array);

    fields.push_back(arrow::field("molecule_id", arrow::uint32()));  // nullable
    arrays.push_back(molecule_id_array);

    // Copy remaining columns
    for (int i = 1; i < valid_rb->num_columns(); ++i) {
        fields.push_back(valid_rb->schema()->field(i));
        arrays.push_back(valid_rb->column(i));
    }

    auto schema = arrow::schema(fields, valid_rb->schema()->metadata());
    auto rb_with_null = arrow::RecordBatch::Make(schema, valid_rb->num_rows(), arrays);

    EXPECT_THROW({
        BondDescriptorBatchFromArrow(rb_with_null);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, BondDescriptorRejectsNullBegin) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoBondDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = BondDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_rows(), 0);

    // Build a record batch with a null begin in the first row
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    fields.push_back(valid_rb->schema()->field(0));
    arrays.push_back(valid_rb->column(0));

    arrow::UInt32Builder begin_builder;
    (void)begin_builder.AppendNull();  // First row: null
    for (std::int64_t i = 1; i < valid_rb->num_rows(); ++i) {
        (void)begin_builder.Append(
            std::dynamic_pointer_cast<arrow::UInt32Array>(valid_rb->column(1))->Value(i));
    }
    std::shared_ptr<arrow::Array> begin_array;
    (void)begin_builder.Finish(&begin_array);

    fields.push_back(arrow::field("begin", arrow::uint32()));  // nullable
    arrays.push_back(begin_array);

    // Copy remaining columns
    for (int i = 2; i < valid_rb->num_columns(); ++i) {
        fields.push_back(valid_rb->schema()->field(i));
        arrays.push_back(valid_rb->column(i));
    }

    auto schema = arrow::schema(fields, valid_rb->schema()->metadata());
    auto rb_with_null = arrow::RecordBatch::Make(schema, valid_rb->num_rows(), arrays);

    EXPECT_THROW({
        BondDescriptorBatchFromArrow(rb_with_null);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, BondDescriptorRejectsNullEnd) {
    auto mol = make_ethane();
    const OEChem::OEMolBase& base = mol;
    std::vector<const OEChem::OEMolBase*> mols{&base};

    KallistoBondDescriptorSource source;
    const auto batch = source.CalculateBatch(mols);
    const auto valid_rb = BondDescriptorBatchToArrow(batch);

    ASSERT_NE(valid_rb, nullptr);
    ASSERT_GT(valid_rb->num_rows(), 0);

    // Build a record batch with a null end in the first row
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    fields.push_back(valid_rb->schema()->field(0));
    arrays.push_back(valid_rb->column(0));
    fields.push_back(valid_rb->schema()->field(1));
    arrays.push_back(valid_rb->column(1));

    arrow::UInt32Builder end_builder;
    (void)end_builder.AppendNull();  // First row: null
    for (std::int64_t i = 1; i < valid_rb->num_rows(); ++i) {
        (void)end_builder.Append(
            std::dynamic_pointer_cast<arrow::UInt32Array>(valid_rb->column(2))->Value(i));
    }
    std::shared_ptr<arrow::Array> end_array;
    (void)end_builder.Finish(&end_array);

    fields.push_back(arrow::field("end", arrow::uint32()));  // nullable
    arrays.push_back(end_array);

    // Copy remaining columns
    for (int i = 3; i < valid_rb->num_columns(); ++i) {
        fields.push_back(valid_rb->schema()->field(i));
        arrays.push_back(valid_rb->column(i));
    }

    auto schema = arrow::schema(fields, valid_rb->schema()->metadata());
    auto rb_with_null = arrow::RecordBatch::Make(schema, valid_rb->num_rows(), arrays);

    EXPECT_THROW({
        BondDescriptorBatchFromArrow(rb_with_null);
    }, std::invalid_argument);
}

TEST(KallistoBatchArrow, AtomDescriptorParquetRoundTrip) {
    // Build batch: [valid, skipped (2D), valid]
    auto mol1 = make_ethane();
    auto mol2 = make_2d_molecule();
    auto mol3 = make_ethane();

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    const OEChem::OEMolBase& base3 = mol3;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2, &base3};

    KallistoAtomDescriptorSource source;
    const auto original_batch = source.CalculateBatch(mols);

    ASSERT_EQ(original_batch.Size(), 3u);
    EXPECT_GT(original_batch.SegmentAtomCount(0), 0u);
    EXPECT_EQ(original_batch.SegmentAtomCount(1), 0u);  // Empty (skipped 2D)
    EXPECT_GT(original_batch.SegmentAtomCount(2), 0u);

    const auto original_atom_count = original_batch.AtomCount();
    EXPECT_GT(original_atom_count, 0u);

    // Write to Parquet and read back
    const std::string path = "/tmp/kallisto_atom_parquet_test.parquet";
    WriteKallistoAtomParquet(original_batch, path);
    const auto reconstructed_batch = ReadKallistoAtomParquet(path);

    // Verify round-trip preserves structure
    ASSERT_EQ(reconstructed_batch.Size(), original_batch.Size());
    EXPECT_EQ(reconstructed_batch.AtomCount(), original_atom_count);

    for (std::size_t mol = 0; mol < original_batch.Size(); ++mol) {
        EXPECT_EQ(reconstructed_batch.SegmentAtomCount(mol),
                  original_batch.SegmentAtomCount(mol))
            << "Segment " << mol << " atom count mismatch";
    }

    // Verify atom indices
    const auto* orig_indices = reinterpret_cast<const std::uint32_t*>(
        original_batch.AtomIndexDataAddress());
    const auto* recon_indices = reinterpret_cast<const std::uint32_t*>(
        reconstructed_batch.AtomIndexDataAddress());

    for (std::size_t i = 0; i < original_atom_count; ++i) {
        EXPECT_EQ(recon_indices[i], orig_indices[i])
            << "Atom index " << i << " mismatch";
    }

    // Verify descriptor values and validity
    const auto& schema = original_batch.Schema();
    for (std::size_t col = 0; col < schema.Size(); ++col) {
        const auto* orig_values = reinterpret_cast<const double*>(
            original_batch.ColumnDataAddress(col));
        const auto* recon_values = reinterpret_cast<const double*>(
            reconstructed_batch.ColumnDataAddress(col));
        const auto* orig_validity = reinterpret_cast<const std::uint8_t*>(
            original_batch.ColumnValidityAddress(col));
        const auto* recon_validity = reinterpret_cast<const std::uint8_t*>(
            reconstructed_batch.ColumnValidityAddress(col));

        for (std::size_t i = 0; i < original_atom_count; ++i) {
            EXPECT_EQ(recon_validity[i], orig_validity[i])
                << "Column " << col << " atom " << i << " validity mismatch";
            if (orig_validity[i] != 0u) {
                EXPECT_DOUBLE_EQ(recon_values[i], orig_values[i])
                    << "Column " << col << " atom " << i << " value mismatch";
            }
        }
    }

    // Verify schema ID is preserved
    EXPECT_EQ(reconstructed_batch.Schema().SchemaId(), original_batch.Schema().SchemaId());
}

TEST(KallistoBatchArrow, BondDescriptorParquetRoundTrip) {
    // Build batch: [valid, skipped (heavy), valid]
    auto mol1 = make_ethane();
    auto mol2 = make_heavy_molecule();
    auto mol3 = make_ethane();

    const OEChem::OEMolBase& base1 = mol1;
    const OEChem::OEMolBase& base2 = mol2;
    const OEChem::OEMolBase& base3 = mol3;
    std::vector<const OEChem::OEMolBase*> mols{&base1, &base2, &base3};

    KallistoBondDescriptorSource source;
    const auto original_batch = source.CalculateBatch(mols);

    ASSERT_EQ(original_batch.Size(), 3u);
    EXPECT_GT(original_batch.SegmentBondCount(0), 0u);
    EXPECT_EQ(original_batch.SegmentBondCount(1), 0u);  // Empty (skipped heavy)
    EXPECT_GT(original_batch.SegmentBondCount(2), 0u);

    const auto original_bond_count = original_batch.BondCount();
    EXPECT_GT(original_bond_count, 0u);

    // Write to Parquet and read back
    const std::string path = "/tmp/kallisto_bond_parquet_test.parquet";
    WriteKallistoBondParquet(original_batch, path);
    const auto reconstructed_batch = ReadKallistoBondParquet(path);

    // Verify round-trip preserves structure
    ASSERT_EQ(reconstructed_batch.Size(), original_batch.Size());
    EXPECT_EQ(reconstructed_batch.BondCount(), original_bond_count);

    for (std::size_t mol = 0; mol < original_batch.Size(); ++mol) {
        EXPECT_EQ(reconstructed_batch.SegmentBondCount(mol),
                  original_batch.SegmentBondCount(mol))
            << "Segment " << mol << " bond count mismatch";
    }

    // Verify bond endpoints
    const auto* orig_begin = reinterpret_cast<const std::uint32_t*>(
        original_batch.BondBeginDataAddress());
    const auto* recon_begin = reinterpret_cast<const std::uint32_t*>(
        reconstructed_batch.BondBeginDataAddress());
    const auto* orig_end = reinterpret_cast<const std::uint32_t*>(
        original_batch.BondEndDataAddress());
    const auto* recon_end = reinterpret_cast<const std::uint32_t*>(
        reconstructed_batch.BondEndDataAddress());

    for (std::size_t i = 0; i < original_bond_count; ++i) {
        EXPECT_EQ(recon_begin[i], orig_begin[i])
            << "Bond " << i << " begin index mismatch";
        EXPECT_EQ(recon_end[i], orig_end[i])
            << "Bond " << i << " end index mismatch";
    }

    // Verify descriptor values and validity
    const auto& schema = original_batch.Schema();
    for (std::size_t col = 0; col < schema.Size(); ++col) {
        const auto* orig_values = reinterpret_cast<const double*>(
            original_batch.ColumnDataAddress(col));
        const auto* recon_values = reinterpret_cast<const double*>(
            reconstructed_batch.ColumnDataAddress(col));
        const auto* orig_validity = reinterpret_cast<const std::uint8_t*>(
            original_batch.ColumnValidityAddress(col));
        const auto* recon_validity = reinterpret_cast<const std::uint8_t*>(
            reconstructed_batch.ColumnValidityAddress(col));

        for (std::size_t i = 0; i < original_bond_count; ++i) {
            EXPECT_EQ(recon_validity[i], orig_validity[i])
                << "Column " << col << " bond " << i << " validity mismatch";
            if (orig_validity[i] != 0u) {
                EXPECT_DOUBLE_EQ(recon_values[i], orig_values[i])
                    << "Column " << col << " bond " << i << " value mismatch";
            }
        }
    }

    // Verify schema ID is preserved
    EXPECT_EQ(reconstructed_batch.Schema().SchemaId(), original_batch.Schema().SchemaId());
}

} // namespace
} // namespace OEFP
