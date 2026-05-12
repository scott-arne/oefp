#include "oefp/openeye.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace OEFP {
namespace {

std::size_t byte_count(std::uint64_t size_bits) {
    return static_cast<std::size_t>((size_bits + 7u) / 8u);
}

std::vector<std::uint64_t> words_from_bytes(
    const unsigned char* data,
    std::uint64_t size_bits) {
    std::vector<std::uint64_t> words(DenseWordCount(size_bits), 0u);
    const auto bytes = byte_count(size_bits);
    for (std::size_t byte_index = 0; byte_index < bytes; ++byte_index) {
        const auto word_index = byte_index / sizeof(std::uint64_t);
        const auto shift = (byte_index % sizeof(std::uint64_t)) * 8u;
        words[word_index] |= static_cast<std::uint64_t>(data[byte_index]) << shift;
    }
    return words;
}

std::vector<unsigned char> bytes_from_words(const OEFP& fp) {
    std::vector<unsigned char> bytes(byte_count(fp.SizeBits()), 0u);
    const auto& words = fp.Words();
    for (std::size_t byte_index = 0; byte_index < bytes.size(); ++byte_index) {
        const auto word_index = byte_index / sizeof(std::uint64_t);
        const auto shift = (byte_index % sizeof(std::uint64_t)) * 8u;
        bytes[byte_index] = static_cast<unsigned char>((words[word_index] >> shift) & 0xFFu);
    }
    return bytes;
}

std::string parameter_summary(const OEGraphSim::OEFPTypeBase* type) {
    if (type == nullptr) {
        return {};
    }

    const OEGraphSim::OEFPTypeParams params(type);
    if (!params.IsValid()) {
        return {};
    }

    std::ostringstream stream;
    stream << "fptype=" << params.GetFPType()
           << ";version=" << params.GetVersion()
           << ";numbits=" << params.GetNumBits()
           << ";mindist=" << params.GetMinDistance()
           << ";maxdist=" << params.GetMaxDistance()
           << ";atomtypes=" << params.GetAtomTypes()
           << ";bondtypes=" << params.GetBondTypes();
    return stream.str();
}

FingerprintSpec openeye_spec_from_fingerprint(const OEGraphSim::OEFingerPrint& fp) {
    FingerprintSpec spec;
    spec.size_bits = fp.GetSize();
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "OpenEye";

    const auto* type = fp.GetFPTypeBase();
    if (type != nullptr) {
        spec.source_type = type->GetFPTypeString();
        spec.source_version = type->GetFPVersionString();
        spec.parameters = parameter_summary(type);
        spec.has_source_type_id = true;
        spec.source_type_id = type->GetFPType();
    } else {
        spec.source_type = "OEFingerPrint";
    }
    return spec;
}

const OEGraphSim::OEFPTypeBase* resolve_openeye_type(const FingerprintSpec& spec) {
    if (spec.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("Only binary OEFP fingerprints can be exported to OEFingerPrint.");
    }
    if (spec.source_name != "OpenEye") {
        throw std::invalid_argument("OEFP spec does not contain OpenEye fingerprint type metadata.");
    }

    if (!spec.source_type.empty()) {
        const auto* type = OEGraphSim::OEGetFPType(spec.source_type);
        if (type != nullptr) {
            return type;
        }
        throw std::invalid_argument("OpenEye fingerprint type metadata could not be resolved.");
    }
    if (spec.has_source_type_id) {
        const auto* type = OEGraphSim::OEGetFPType(spec.source_type_id);
        if (type != nullptr) {
            return type;
        }
    }
    throw std::invalid_argument("OpenEye fingerprint type metadata could not be resolved.");
}

} // namespace

OEFP FromOEFingerPrint(const OEGraphSim::OEFingerPrint& fp) {
    const auto spec = openeye_spec_from_fingerprint(fp);
    return OEFP(spec, words_from_bytes(fp.GetData(), spec.size_bits));
}

OEGraphSim::OEFingerPrint ToOEFingerPrint(const OEFP& fp) {
    if (fp.SizeBits() > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument("OEFingerPrint cannot store this many bits.");
    }

    const auto* type = resolve_openeye_type(fp.Spec());
    OEGraphSim::OEFingerPrint out;
    const auto bytes = bytes_from_words(fp);
    if (fp.SizeBits() == 0) {
        out.SetSize(0);
    } else {
        out.SetData(bytes.data(), static_cast<unsigned int>(fp.SizeBits()));
    }
    out.SetFPTypeBase(type);
    return out;
}

} // namespace OEFP
