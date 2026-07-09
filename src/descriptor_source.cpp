#include "oefp/descriptor_source.h"

#include "oefp/molecular_properties.h"
#include "oefp/mordred.h"

#include <array>
#include <cstdint>
#include <string>

#include <oechem.h>
#include <oemolprop.h>
#include <oeplatform.h>

namespace OEFP {

DescriptorSet DescriptorSource::Compute(const OEChem::OEMolBase& mol,
                                        ComputeContext&,
                                        const ColumnRequest&) const {
    return Compute(mol);
}

DescriptorSet DescriptorSource::Compute(const OEChem::OEMolBase& mol,
                                        ComputeContext& ctx) const {
    return Compute(mol, ctx, ColumnRequest::All());
}

std::shared_ptr<const DescriptorSchema> MordredDescriptorSource::Schema() const {
    return MordredDescriptorSchema();
}

DescriptorSet MordredDescriptorSource::Compute(const OEChem::OEMolBase& mol) const {
    return MakeMordredDescriptors(mol);
}

DescriptorSet MordredDescriptorSource::Compute(const OEChem::OEMolBase& mol,
                                               ComputeContext& ctx,
                                               const ColumnRequest& request) const {
    return MakeMordredDescriptors(mol, ctx, request);
}

namespace {

struct StaticOpenEyeDefinition {
    const char* name;
    DescriptorValueKind value_kind;
    const char* canonical_id;
};

/// The seven tagged definitions below share their computation with the matching
/// Mordred column (via the shared molecular-property helpers for weights and
/// counts, and the identical OpenEye calls with Mordred's preparation for
/// TopoPSA/HBD/HBA), so their ``canonical_id`` marks them identical by
/// construction. Changing any of these implementations, their curated
/// ``canonical_id``, or the shared helpers requires re-verifying the identity
/// pairings in the canonical-identity audit test. The four untagged
/// definitions are genuinely OpenEye-unique and intentionally carry no
/// ``canonical_id`` so they never deduplicate.
constexpr std::array<StaticOpenEyeDefinition, 11> kOpenEyeDefinitions{{
    {"MolecularWeight", DescriptorValueKind::Float, "quantity:exact_molecular_weight"},
    {"AverageMolecularWeight", DescriptorValueKind::Float, "quantity:average_molecular_weight"},
    {"HeavyAtomCount", DescriptorValueKind::Int, "quantity:heavy_atom_count"},
    {"TotalAtomCount", DescriptorValueKind::Int, "quantity:total_atom_count"},
    {"TopologicalPSA", DescriptorValueKind::Float, "quantity:topological_psa"},
    {"LipinskiHBD", DescriptorValueKind::Int, "quantity:num_hbond_donors_lipinski"},
    {"HBA", DescriptorValueKind::Int, "quantity:num_hbond_acceptors"},
    {"XLogP", DescriptorValueKind::Float, ""},
    {"FractionCsp3", DescriptorValueKind::Float, ""},
    {"AromaticRingCount", DescriptorValueKind::Int, ""},
    {"RotatableBondCount", DescriptorValueKind::Int, ""},
}};

} // namespace

std::shared_ptr<const DescriptorSchema> OpenEyePropertyDescriptorSource::Schema() const {
    static const std::shared_ptr<const DescriptorSchema> schema = [] {
        DescriptorSchemaBuilder builder;
        const std::string source_version = OEPlatform::OEToolkitsGetRelease();
        for (const auto& item : kOpenEyeDefinitions) {
            DescriptorDefinition definition;
            definition.name = item.name;
            definition.value_kind = item.value_kind;
            definition.group = "openeye:properties";
            definition.source_name = "OpenEye";
            definition.source_version = source_version;
            definition.canonical_id = item.canonical_id;
            builder.Add(std::move(definition));
        }
        return builder.Build();
    }();
    return schema;
}

DescriptorSet OpenEyePropertyDescriptorSource::Compute(const OEChem::OEMolBase& mol) const {
    // Delegate through a fresh per-molecule context so there is a single
    // implementation; the context supplies the shared ring-perceived molecule.
    ComputeContext ctx(mol);
    return Compute(ctx.NormalizedMol(), ctx, ColumnRequest::All());
}

DescriptorSet OpenEyePropertyDescriptorSource::Compute(const OEChem::OEMolBase& mol,
                                                       ComputeContext& ctx,
                                                       const ColumnRequest& request) const {
    const auto schema = Schema();
    DescriptorSetBuilder builder(schema);

    // Each column computes (and emits) only when the request wants it. Pruning
    // changes WHICH columns run, never HOW a column is computed, so a requested
    // column's value is byte-identical to the All() value. The underlying
    // OpenEye/shared-helper calls are inside the guards so skipping a column
    // skips its work too.
    const auto wants = [&](const char* name) {
        return request.Wants(schema->IndexOf(name));
    };

    // Tagged columns identical by construction with Mordred. Weights and atom
    // counts reuse the shared helpers on the input molecule.
    if (wants("MolecularWeight")) {
        builder.Set("MolecularWeight", DescriptorValue::Float(ExactMolecularWeight(mol)));
    }
    if (wants("AverageMolecularWeight")) {
        builder.Set("AverageMolecularWeight", DescriptorValue::Float(AverageMolecularWeight(mol)));
    }
    if (wants("HeavyAtomCount")) {
        builder.Set(
            "HeavyAtomCount",
            DescriptorValue::Int(static_cast<std::int64_t>(HeavyAtomCount(mol))));
    }
    if (wants("TotalAtomCount")) {
        builder.Set(
            "TotalAtomCount",
            DescriptorValue::Int(static_cast<std::int64_t>(TotalAtomCount(mol))));
    }

    // TopoPSA/HBD/HBA pull the ring-perceived working molecule from the shared
    // context so the OpenEye calls see the same perceived graph Mordred feeds
    // them. The context prep (OEGraphMol copy + OEFindRingAtomsAndBonds +
    // OEAssignHybridization) is byte-identical to the former inline block, so
    // these tagged values are unchanged. The fetch is conditional: it happens
    // only when at least one ring-perception-consuming column is wanted, so
    // pruning to columns that do not need it avoids the shared computation.
    const bool wants_topological_psa = wants("TopologicalPSA");
    const bool wants_lipinski_hbd = wants("LipinskiHBD");
    const bool wants_hba = wants("HBA");
    if (wants_topological_psa || wants_lipinski_hbd || wants_hba) {
        const OEChem::OEGraphMol& working_mol = ctx.RingPerceivedMol();
        if (wants_topological_psa) {
            float psa = 0.0f;
            if (OEMolProp::OEGet2dPSA(working_mol, psa, nullptr, true)) {
                builder.Set(
                    "TopologicalPSA",
                    DescriptorValue::Float(RoundTopologicalPsa(static_cast<double>(psa))));
            }
        }
        if (wants_lipinski_hbd) {
            builder.Set(
                "LipinskiHBD",
                DescriptorValue::Int(
                    static_cast<std::int64_t>(OEMolProp::OEGetLipinskiDonorCount(working_mol))));
        }
        if (wants_hba) {
            builder.Set(
                "HBA",
                DescriptorValue::Int(
                    static_cast<std::int64_t>(OEMolProp::OEGetHBondAcceptorCount(working_mol))));
        }
    }

    // Untagged, genuinely OpenEye-unique columns computed directly on the input
    // molecule (matching the OpenEye-toolkit conformance oracle).
    if (wants("XLogP")) {
        float xlogp = 0.0f;
        if (OEMolProp::OEGetXLogP(mol, xlogp)) {
            builder.Set("XLogP", DescriptorValue::Float(static_cast<double>(xlogp)));
        }
    }
    if (wants("FractionCsp3")) {
        builder.Set(
            "FractionCsp3",
            DescriptorValue::Float(static_cast<double>(OEMolProp::OEGetFractionCsp3(mol))));
    }
    if (wants("AromaticRingCount")) {
        builder.Set(
            "AromaticRingCount",
            DescriptorValue::Int(
                static_cast<std::int64_t>(OEMolProp::OEGetAromaticRingCount(mol))));
    }
    if (wants("RotatableBondCount")) {
        builder.Set(
            "RotatableBondCount",
            DescriptorValue::Int(
                static_cast<std::int64_t>(OEMolProp::OEGetRotatableBondCount(mol))));
    }

    return builder.Build();
}

} // namespace OEFP
