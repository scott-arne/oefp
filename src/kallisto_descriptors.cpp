#include "oefp/kallisto_descriptors.h"
#include "oefp/kallisto_data.h"
#include "oefp/atom_descriptor.h"

#include <oechem.h>

#include <cmath>
#include <stdexcept>

namespace OEFP {

std::vector<double> coordination_numbers(
    const KallistoGeometryContext& ctx,
    const std::string& cntype
) {
    // Return empty for ineligible contexts (2D, Z>86, coord-failure)
    if (!ctx.Eligible()) {
        return std::vector<double>{};
    }

    const auto& atomic_numbers = ctx.AtomicNumbers();
    const auto& coords = ctx.CoordsBohr();
    const std::size_t nat = atomic_numbers.size();

    std::vector<double> cns(nat, 0.0);

    constexpr double threshold = 800.0;  // Bohr² squared-distance cutoff

    if (cntype == "exp") {
        constexpr double k1 = 16.0;
        for (std::size_t i = 0; i < nat; ++i) {
            const int zi = atomic_numbers[i];
            const int ia = zi - 1;  // Convert to 0-based index
            const auto& ci = coords[i];

            for (std::size_t j = 0; j < nat; ++j) {
                if (i == j) continue;

                const auto& cj = coords[j];
                const double dx = cj[0] - ci[0];
                const double dy = cj[1] - ci[1];
                const double dz = cj[2] - ci[2];
                const double r_squared = dx * dx + dy * dy + dz * dz;

                if (r_squared > threshold) continue;

                const int zj = atomic_numbers[j];
                const int ja = zj - 1;
                const double r = std::sqrt(r_squared);
                const double rco = kallisto::COVALENT_RADIUS[ia] + kallisto::COVALENT_RADIUS[ja];
                const double rr = rco / r;
                const double k = -k1 * (rr - 1.0);
                const double damp = 1.0 / (1.0 + std::exp(k));
                cns[i] += damp;
            }
        }
    } else if (cntype == "erf") {
        constexpr double kn = 7.50;
        for (std::size_t i = 0; i < nat; ++i) {
            const int zi = atomic_numbers[i];
            const int ia = zi - 1;
            const auto& ci = coords[i];

            for (std::size_t j = 0; j < nat; ++j) {
                if (i == j) continue;

                const int zj = atomic_numbers[j];
                const int ja = zj - 1;
                const auto& cj = coords[j];
                const double dx = cj[0] - ci[0];
                const double dy = cj[1] - ci[1];
                const double dz = cj[2] - ci[2];
                const double r_squared = dx * dx + dy * dy + dz * dz;

                if (r_squared > threshold) continue;

                const double r = std::sqrt(r_squared);
                const double rco = kallisto::COVALENT_RADIUS[ia] + kallisto::COVALENT_RADIUS[ja];
                const double damp = 0.5 * (1.0 + std::erf(-kn * (r - rco) / rco));
                cns[i] += damp;
            }
        }
    } else if (cntype == "cov") {
        // Fitted to match Wiberg bond orders of diatomic molecules
        constexpr double k4 = 4.10451;
        constexpr double k5 = 19.08857;
        constexpr double k6 = 2.0 * 11.28174 * 11.28174;
        constexpr double kn = 7.50;

        for (std::size_t i = 0; i < nat; ++i) {
            const int zi = atomic_numbers[i];
            const int ia = zi - 1;
            const auto& ci = coords[i];

            for (std::size_t j = 0; j < nat; ++j) {
                if (i == j) continue;

                const int zj = atomic_numbers[j];
                const int ja = zj - 1;
                const auto& cj = coords[j];
                const double dx = cj[0] - ci[0];
                const double dy = cj[1] - ci[1];
                const double dz = cj[2] - ci[2];
                const double r_squared = dx * dx + dy * dy + dz * dz;

                if (r_squared > threshold) continue;

                const double r = std::sqrt(r_squared);
                const double rco = kallisto::COVALENT_RADIUS[ia] + kallisto::COVALENT_RADIUS[ja];
                const double eni = kallisto::PAULING_EN[ia];
                const double enj = kallisto::PAULING_EN[ja];
                const double en_diff = std::abs(eni - enj);
                const double den = k4 * std::exp(-((en_diff + k5) * (en_diff + k5)) / k6);
                const double damp = den * 0.5 * (1.0 + std::erf(-kn * (r - rco) / rco));
                cns[i] += damp;
            }
        }
    } else {
        throw std::invalid_argument("Unknown cntype: " + cntype + " (expected erf, cov, or exp)");
    }

    return cns;
}

std::vector<double> proximity_shells(
    const KallistoGeometryContext& ctx,
    std::pair<int, int> size
) {
    // Return empty for ineligible contexts (2D, Z>86, coord-failure)
    if (!ctx.Eligible()) {
        return std::vector<double>{};
    }

    const auto& atomic_numbers = ctx.AtomicNumbers();
    const auto& coords = ctx.CoordsBohr();
    const std::size_t nat = atomic_numbers.size();

    std::vector<double> prox1(nat, 0.0);
    std::vector<double> prox2(nat, 0.0);

    // Fitted to match Wiberg bond orders of diatomic molecules
    constexpr double k4 = 4.10451;
    constexpr double k5 = 19.08857;
    constexpr double k6 = 2.0 * 11.28174 * 11.28174;
    constexpr double kn = 7.50;
    constexpr double threshold = 800.0;  // Bohr² squared-distance cutoff

    const int scale1 = size.first;
    const int scale2 = size.second;

    for (std::size_t i = 0; i < nat; ++i) {
        const int zi = atomic_numbers[i];
        const int ia = zi - 1;
        const auto& ci = coords[i];

        for (std::size_t j = 0; j < nat; ++j) {
            if (i == j) continue;

            const int zj = atomic_numbers[j];
            const int ja = zj - 1;
            const auto& cj = coords[j];
            const double dx = cj[0] - ci[0];
            const double dy = cj[1] - ci[1];
            const double dz = cj[2] - ci[2];
            const double r_squared = dx * dx + dy * dy + dz * dz;

            if (r_squared > threshold) continue;

            const double r = std::sqrt(r_squared);
            const double eni = kallisto::PAULING_EN[ia];
            const double enj = kallisto::PAULING_EN[ja];
            const double en_diff = std::abs(eni - enj);
            const double den = k4 * std::exp(-((en_diff + k5) * (en_diff + k5)) / k6);

            // Smaller border (shell 1)
            {
                const double rco = scale1 * (kallisto::COVALENT_RADIUS[ia] + kallisto::COVALENT_RADIUS[ja]);
                const double damp = den * 0.5 * (1.0 + std::erf(-kn * (r - rco) / rco));
                prox1[i] += damp;
            }

            // Larger border (shell 2)
            {
                const double rco = scale2 * (kallisto::COVALENT_RADIUS[ia] + kallisto::COVALENT_RADIUS[ja]);
                const double damp = den * 0.5 * (1.0 + std::erf(-kn * (r - rco) / rco));
                prox2[i] += damp;
            }
        }
    }

    // Return the difference (shell 2 - shell 1)
    std::vector<double> result(nat);
    for (std::size_t i = 0; i < nat; ++i) {
        result[i] = prox2[i] - prox1[i];
    }
    return result;
}

std::vector<double> eeq_charges(const KallistoGeometryContext& ctx) {
    // Return empty for ineligible contexts
    if (!ctx.Eligible()) {
        return std::vector<double>{};
    }

    const auto& atomic_numbers = ctx.AtomicNumbers();
    const auto& coords = ctx.CoordsBohr();
    const auto& covcn = ctx.CovalentCoordinationNumbers();
    const std::size_t nat = atomic_numbers.size();

    // Lagrange space is +1 in dimensionality
    const std::size_t m = nat + 1;

    // Build parameter arrays
    const double sqrt2pi = std::sqrt(2.0 / M_PI);
    std::vector<double> alpha(nat);
    std::vector<double> gam(nat);
    std::vector<double> xi(nat);
    std::vector<double> kappa(nat);

    for (std::size_t i = 0; i < nat; ++i) {
        const int zi = atomic_numbers[i];
        const int ia = zi - 1;  // 0-based index
        alpha[i] = kallisto::EEQ_ALP[ia] * kallisto::EEQ_ALP[ia];  // Square it
        gam[i] = kallisto::EEQ_GAMM[ia];
        xi[i] = kallisto::EEQ_EN[ia];
        kappa[i] = kallisto::EEQ_CNFAK[ia];
    }

    // Build A matrix (row-major)
    std::vector<double> A(m * m, 0.0);

    for (std::size_t i = 0; i < nat; ++i) {
        const auto& ci = coords[i];

        // Diagonal: A[i][i] = gam[i] + sqrt2pi / sqrt(alpha[i])
        A[i * m + i] = gam[i] + sqrt2pi / std::sqrt(alpha[i]);

        for (std::size_t j = 0; j < nat; ++j) {
            if (i == j) continue;

            const auto& cj = coords[j];
            const double dx = cj[0] - ci[0];
            const double dy = cj[1] - ci[1];
            const double dz = cj[2] - ci[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

            // gamij = 1 / sqrt(alpha_i + alpha_j)
            const double gamij = 1.0 / std::sqrt(alpha[i] + alpha[j]);

            // Off-diagonal: A[i][j] = erf(gamij * r) / r
            const double val = std::erf(gamij * r) / r;
            A[i * m + j] = val;
        }
    }

    // Lagrange augmentation: last row and column all 1, corner 0
    for (std::size_t i = 0; i < nat; ++i) {
        A[i * m + nat] = 1.0;      // Column nat
        A[nat * m + i] = 1.0;      // Row nat
    }
    A[nat * m + nat] = 0.0;        // Corner

    // Build X vector
    std::vector<double> X(m, 0.0);
    for (std::size_t i = 0; i < nat; ++i) {
        X[i] = -xi[i] + kappa[i] * std::sqrt(covcn[i]);
    }
    X[nat] = static_cast<double>(ctx.Charge());

    // Solve A q = X
    auto solution = solve_dense_linear(A, X, m);

    // If singular, return empty (degenerate case)
    if (!solution.has_value()) {
        return std::vector<double>{};
    }

    // Drop the Lagrange multiplier (last element)
    std::vector<double> qs = solution.value();
    qs.resize(nat);
    return qs;
}

std::shared_ptr<const DescriptorSchema> KallistoAtomDescriptorSchema() {
    static const std::shared_ptr<const DescriptorSchema> schema = [] {
        // Table of {name, units, description} for Task 5-6 columns
        // Later tasks (7-9) will append to this table
        struct ColumnDef {
            const char* name;
            const char* units;
            const char* description;
        };

        static constexpr ColumnDef kColumns[] = {
            {"cn_erf", "", "Error-function coordination number"},
            {"cn_cov", "", "Covalent coordination number (electronegativity-weighted)"},
            {"cn_exp", "", "Exponential coordination number"},
            {"prox", "", "Proximity shell difference (scale 2-3)"},
            {"eeq", "e", "EEQ atomic partial charge (electronegativity equilibration)"},
        };

        std::vector<DescriptorDefinition> definitions;
        definitions.reserve(sizeof(kColumns) / sizeof(kColumns[0]));

        for (const auto& col : kColumns) {
            DescriptorDefinition def;
            def.name = col.name;
            def.value_kind = DescriptorValueKind::Float;
            def.group = "kallisto";
            def.source_name = "kallisto";
            def.source_type = "geometric";
            def.source_version = "kallisto-1.0.10";
            def.units = col.units;
            def.description = col.description;
            def.prerequisites = kDescriptorPrerequisiteCoordinates3D;
            definitions.push_back(std::move(def));
        }

        return std::make_shared<const DescriptorSchema>(std::move(definitions));
    }();
    return schema;
}

AtomDescriptorSet MakeKallistoAtomDescriptors(
    const OEChem::OEMolBase& mol,
    std::optional<int> charge
) {
    auto schema = KallistoAtomDescriptorSchema();

    // Build geometry context
    KallistoGeometryContext ctx(mol, charge);

    // Ineligible -> empty set
    if (!ctx.Eligible()) {
        return AtomDescriptorSet::Empty(schema);
    }

    const std::size_t atom_count = ctx.AtomCount();

    // Compute all five columns
    auto cn_erf_vals = coordination_numbers(ctx, "erf");
    auto cn_cov_vals = coordination_numbers(ctx, "cov");
    auto cn_exp_vals = coordination_numbers(ctx, "exp");
    auto prox_vals = proximity_shells(ctx, {2, 3});
    auto eeq_vals = ctx.EeqCharges();

    // Build columns as [column][atom] of std::optional<double>
    // Column order MUST match schema: cn_erf, cn_cov, cn_exp, prox, eeq
    std::vector<std::vector<std::optional<double>>> columns(5);
    columns[0].reserve(atom_count);
    columns[1].reserve(atom_count);
    columns[2].reserve(atom_count);
    columns[3].reserve(atom_count);
    columns[4].reserve(atom_count);

    // Guard against singular solves: kernel functions (like eeq_charges) return
    // an empty vector on degenerate cases. If a kernel result size doesn't match
    // atom_count, emit that column as all-missing instead of indexing out-of-bounds.
    const bool cn_erf_ok = cn_erf_vals.size() == atom_count;
    const bool cn_cov_ok = cn_cov_vals.size() == atom_count;
    const bool cn_exp_ok = cn_exp_vals.size() == atom_count;
    const bool prox_ok = prox_vals.size() == atom_count;
    const bool eeq_ok = eeq_vals.size() == atom_count;

    for (std::size_t i = 0; i < atom_count; ++i) {
        columns[0].push_back(cn_erf_ok ? std::optional<double>(cn_erf_vals[i]) : std::nullopt);
        columns[1].push_back(cn_cov_ok ? std::optional<double>(cn_cov_vals[i]) : std::nullopt);
        columns[2].push_back(cn_exp_ok ? std::optional<double>(cn_exp_vals[i]) : std::nullopt);
        columns[3].push_back(prox_ok ? std::optional<double>(prox_vals[i]) : std::nullopt);
        columns[4].push_back(eeq_ok ? std::optional<double>(eeq_vals[i]) : std::nullopt);
    }

    // Get OpenEye atom indices from context (guaranteed aligned with geometry)
    std::vector<std::uint32_t> atom_indices = ctx.AtomIndices();

    return AtomDescriptorSet(schema, std::move(atom_indices), std::move(columns));
}

KallistoAtomDescriptorSource::KallistoAtomDescriptorSource(std::optional<int> charge)
    : charge_(charge) {}

std::shared_ptr<const DescriptorSchema> KallistoAtomDescriptorSource::Schema() const {
    return KallistoAtomDescriptorSchema();
}

AtomDescriptorSet KallistoAtomDescriptorSource::Compute(const OEChem::OEMolBase& mol) const {
    return MakeKallistoAtomDescriptors(mol, charge_);
}

AtomDescriptorBatch MakeKallistoAtomDescriptorBatch(
    const OEChem::OEMolBase& mol,
    std::optional<int> charge
) {
    auto set = MakeKallistoAtomDescriptors(mol, charge);
    auto batch = AtomDescriptorBatch::Empty(KallistoAtomDescriptorSchema());
    batch.Append(set);
    return batch;
}

} // namespace OEFP
