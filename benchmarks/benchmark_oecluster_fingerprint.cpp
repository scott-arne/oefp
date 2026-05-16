#include "oefp/batch.h"
#include "oefp/compare.h"
#include "oefp/metric.h"
#include "oefp/openeye.h"

#include "oecluster/CDist.h"
#include "oecluster/PDist.h"
#include "oecluster/StorageBackend.h"
#include "oecluster/metrics/FingerprintMetric.h"

#include <oechem.h>
#include <oegraphsim.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> seed_smiles() {
    return {
        "c1ccccc1",
        "c1ccc(O)cc1",
        "c1ccncc1",
        "CCCCCCCC",
        "CCO",
        "CCN",
        "CC(=O)O",
        "c1ccc(Cl)cc1",
    };
}

std::vector<OEChem::OEGraphMol> build_molecules(std::size_t count) {
    const auto seeds = seed_smiles();
    std::vector<OEChem::OEGraphMol> mols;
    mols.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        OEChem::OEGraphMol mol;
        if (!OEChem::OESmilesToMol(mol, seeds[index % seeds.size()])) {
            throw std::runtime_error("OESmilesToMol failed.");
        }
        mols.push_back(mol);
    }
    return mols;
}

std::vector<OEChem::OEMolBase*> mol_ptrs(std::vector<OEChem::OEGraphMol>& mols) {
    std::vector<OEChem::OEMolBase*> ptrs;
    ptrs.reserve(mols.size());
    for (auto& mol : mols) {
        ptrs.push_back(&static_cast<OEChem::OEMolBase&>(mol));
    }
    return ptrs;
}

OEFP::OEFPBatch build_oefp_batch(const std::vector<OEChem::OEGraphMol>& mols) {
    std::vector<OEFP::OEFP> fps;
    fps.reserve(mols.size());
    for (const auto& mol : mols) {
        OEGraphSim::OEFingerPrint oe_fp;
        if (!OEGraphSim::OEMakeCircularFP(
                oe_fp,
                mol,
                2048u,
                0u,
                2u,
                OEGraphSim::OEFPAtomType::DefaultCircularAtom,
                OEGraphSim::OEFPBondType::DefaultCircularBond)) {
            throw std::runtime_error("OEMakeCircularFP failed.");
        }
        fps.push_back(OEFP::FromOEFingerPrint(oe_fp));
    }
    return OEFP::OEFPBatch::FromFingerprints(fps);
}

template <typename Func>
double elapsed_seconds(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

void assert_close(const std::vector<double>& a, const double* b, std::size_t size) {
    if (a.size() != size) {
        throw std::runtime_error("OEFP and oecluster result sizes differ.");
    }
    for (std::size_t index = 0; index < size; ++index) {
        if (std::fabs(a[index] - b[index]) > 1e-7) {
            throw std::runtime_error("OEFP and oecluster results differ.");
        }
    }
}

void assert_close(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("OEFP and oecluster result sizes differ.");
    }
    for (std::size_t index = 0; index < a.size(); ++index) {
        if (std::fabs(a[index] - b[index]) > 1e-7) {
            throw std::runtime_error("OEFP and oecluster results differ.");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const auto count = argc > 1 ? static_cast<std::size_t>(std::stoul(argv[1])) : 512u;
    const auto threads = argc > 2 ? static_cast<std::size_t>(std::stoul(argv[2])) : 0u;
    const auto chunk_size = argc > 3 ? static_cast<std::size_t>(std::stoul(argv[3])) : 256u;

    auto mols = build_molecules(count);
    auto ptrs = mol_ptrs(mols);
    const auto batch = build_oefp_batch(mols);
    const auto metric = OEFP::Metric::Jaccard();
    OEFP::BatchKernelOptions oe_options;
    oe_options.num_threads = threads;
    oe_options.chunk_size = chunk_size;

    OECluster::FingerprintOptions fp_options;
    fp_options.fp_type = "circular";
    fp_options.numbits = 2048u;
    fp_options.min_distance = 0u;
    fp_options.max_distance = 2u;
    fp_options.similarity_func = "tanimoto";
    fp_options.similarity = false;

    OECluster::PDistOptions pd_options;
    pd_options.num_threads = threads;
    pd_options.chunk_size = chunk_size;

    OECluster::FingerprintMetric pd_metric(ptrs, fp_options);
    OECluster::DenseStorage storage(count);

    const double oe_pdist_time = elapsed_seconds([&]() {
        OECluster::pdist(pd_metric, storage, pd_options);
    });
    std::vector<double> oe_fp_pdist;
    const double fp_pdist_time = elapsed_seconds([&]() {
        oe_fp_pdist = OEFP::PDist(batch, metric, oe_options);
    });
    assert_close(oe_fp_pdist, storage.Data(), storage.NumPairs());

    const std::size_t n_a = count / 2u;
    std::vector<OEChem::OEGraphMol> mols_a(
        mols.begin(),
        mols.begin() + static_cast<long>(n_a));
    std::vector<OEChem::OEGraphMol> mols_b(
        mols.begin() + static_cast<long>(n_a),
        mols.end());
    const auto batch_a = build_oefp_batch(mols_a);
    const auto batch_b = build_oefp_batch(mols_b);
    std::vector<OEChem::OEMolBase*> cd_ptrs;
    cd_ptrs.reserve(mols.size());
    for (auto& mol : mols_a) {
        cd_ptrs.push_back(&static_cast<OEChem::OEMolBase&>(mol));
    }
    for (auto& mol : mols_b) {
        cd_ptrs.push_back(&static_cast<OEChem::OEMolBase&>(mol));
    }

    OECluster::FingerprintMetric cd_metric(cd_ptrs, fp_options);
    OECluster::CDistOptions cd_options;
    cd_options.num_threads = threads;
    cd_options.chunk_size = chunk_size;
    std::vector<double> oe_cd_out(n_a * (count - n_a), 0.0);

    const double oe_cdist_time = elapsed_seconds([&]() {
        OECluster::cdist(cd_metric, n_a, oe_cd_out.data(), cd_options);
    });
    std::vector<double> oe_fp_cd;
    const double fp_cdist_time = elapsed_seconds([&]() {
        oe_fp_cd = OEFP::CDist(batch_a, batch_b, metric, oe_options);
    });
    assert_close(oe_fp_cd, oe_cd_out);

    std::cout << "oecluster pdist: " << oe_pdist_time << "s\n";
    std::cout << "oefp pdist:      " << fp_pdist_time << "s\n";
    std::cout << "oecluster cdist: " << oe_cdist_time << "s\n";
    std::cout << "oefp cdist:      " << fp_cdist_time << "s\n";

    if (fp_pdist_time > oe_pdist_time * 1.05) {
        throw std::runtime_error("OEFP pdist is more than 5% slower than oecluster.");
    }
    if (fp_cdist_time > oe_cdist_time * 1.05) {
        throw std::runtime_error("OEFP cdist is more than 5% slower than oecluster.");
    }
    return 0;
}
