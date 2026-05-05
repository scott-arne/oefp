#include "oefp/batch.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace OEFP {

OEFPBatch::OEFPBatch(FingerprintSpec spec)
    : spec_(std::move(spec)),
      words_per_fingerprint_(DenseWordCount(spec_.size_bits)) {
    if (spec_.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("OEFPBatch supports binary fingerprints only.");
    }
}

OEFPBatch OEFPBatch::FromFingerprints(const std::vector<OEFP>& fingerprints) {
    if (fingerprints.empty()) {
        return OEFPBatch();
    }

    OEFPBatch batch(fingerprints.front().Spec());
    batch.ReserveRows(fingerprints.size());
    for (const auto& fp : fingerprints) {
        batch.Append(fp);
    }
    return batch;
}

void OEFPBatch::Append(const OEFP& fp) {
    ValidateFingerprint(fp);

    const auto popcount = fp.CountOnBits();
    if (popcount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Fingerprint popcount exceeds uint32 storage.");
    }

    const auto& fp_words = fp.Words();
    ReserveRows(Size() + 1);
    words_.insert(words_.end(), fp_words.begin(), fp_words.end());
    popcounts_.push_back(static_cast<std::uint32_t>(popcount));
}

const FingerprintSpec& OEFPBatch::Spec() const {
    return spec_;
}

std::size_t OEFPBatch::Size() const {
    return popcounts_.size();
}

std::uint64_t OEFPBatch::SizeBits() const {
    return spec_.size_bits;
}

std::size_t OEFPBatch::WordsPerFingerprint() const {
    return words_per_fingerprint_;
}

std::size_t OEFPBatch::WordCount() const {
    return words_.size();
}

const std::uint64_t* OEFPBatch::RowWords(std::size_t row) const {
    CheckRowIndex(row);
    return words_.data() + row * words_per_fingerprint_;
}

const std::vector<std::uint64_t>& OEFPBatch::Words() const {
    return words_;
}

const std::vector<std::uint32_t>& OEFPBatch::PopCounts() const {
    return popcounts_;
}

std::uint32_t OEFPBatch::PopCount(std::size_t row) const {
    CheckRowIndex(row);
    return popcounts_[row];
}

const std::uint64_t* OEFPBatch::WordData() const {
    if (words_.empty()) {
        return nullptr;
    }
    return words_.data();
}

const std::uint32_t* OEFPBatch::PopCountData() const {
    if (popcounts_.empty()) {
        return nullptr;
    }
    return popcounts_.data();
}

std::uintptr_t OEFPBatch::WordDataAddress() const {
    const auto* data = WordData();
    if (data == nullptr) {
        return 0;
    }
    return reinterpret_cast<std::uintptr_t>(data);
}

std::uintptr_t OEFPBatch::PopCountDataAddress() const {
    const auto* data = PopCountData();
    if (data == nullptr) {
        return 0;
    }
    return reinterpret_cast<std::uintptr_t>(data);
}

void OEFPBatch::ReserveRows(std::size_t row_count) {
    if (words_per_fingerprint_ != 0
        && row_count > std::numeric_limits<std::size_t>::max() / words_per_fingerprint_) {
        throw std::invalid_argument("Requested batch word capacity is too large.");
    }

    words_.reserve(row_count * words_per_fingerprint_);
    popcounts_.reserve(row_count);
}

void OEFPBatch::ValidateFingerprint(const OEFP& fp) const {
    if (fp.Spec() != spec_) {
        throw std::invalid_argument("Fingerprint spec does not match batch spec.");
    }
    if (fp.WordCount() != words_per_fingerprint_) {
        throw std::invalid_argument("Fingerprint word count does not match batch row width.");
    }
    if (words_per_fingerprint_ == 0) {
        throw std::invalid_argument("Cannot append zero-width fingerprints to a batch.");
    }
}

void OEFPBatch::CheckRowIndex(std::size_t row) const {
    if (row >= Size()) {
        throw std::out_of_range("Batch row index is out of range.");
    }
}

} // namespace OEFP
