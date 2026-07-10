#include "oefp/kallisto_descriptors.h"
#include "oefp/kallisto_data.h"

#include <cmath>
#include <stdexcept>

namespace OEFP {

std::vector<double> coordination_numbers(
    const KallistoGeometryContext& ctx,
    const std::string& cntype
) {
    const std::size_t nat = ctx.AtomCount();
    const auto& atomic_numbers = ctx.AtomicNumbers();
    const auto& coords = ctx.CoordsBohr();

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
    const std::size_t nat = ctx.AtomCount();
    const auto& atomic_numbers = ctx.AtomicNumbers();
    const auto& coords = ctx.CoordsBohr();

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

std::shared_ptr<const DescriptorSchema> KallistoAtomDescriptorSchema() {
    static const std::shared_ptr<const DescriptorSchema> schema = [] {
        // Table of {name, units, description} for Task 5 columns
        // Later tasks will append to this table
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

} // namespace OEFP
