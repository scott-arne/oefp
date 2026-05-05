#include "oefp/metric.h"

#include <stdexcept>
#include <string>

namespace OEFP {
namespace {

void validate_tversky_parameter(double value, const char* name) {
    if (!(value >= 0.0 && value <= 1.0)) {
        throw std::invalid_argument(std::string("Tversky ") + name + " must be in [0.0, 1.0].");
    }
}

} // namespace

Metric Metric::Tanimoto(MetricMode mode) {
    return Metric(MetricKind::Tanimoto, mode, 1.0, 1.0);
}

Metric Metric::Jaccard(MetricMode mode) {
    return Metric(MetricKind::Jaccard, mode, 1.0, 1.0);
}

Metric Metric::Tversky(double alpha, double beta, MetricMode mode) {
    validate_tversky_parameter(alpha, "alpha");
    validate_tversky_parameter(beta, "beta");
    return Metric(MetricKind::Tversky, mode, alpha, beta);
}

Metric Metric::Dice(MetricMode mode) {
    return Metric(MetricKind::Dice, mode, 1.0, 1.0);
}

Metric Metric::Cosine(MetricMode mode) {
    return Metric(MetricKind::Cosine, mode, 1.0, 1.0);
}

Metric Metric::Manhattan(MetricMode mode) {
    if (mode == MetricMode::Similarity) {
        throw std::invalid_argument("Manhattan similarity is not defined for binary fingerprints.");
    }
    return Metric(MetricKind::Manhattan, mode, 1.0, 1.0);
}

MetricKind Metric::Kind() const {
    return kind_;
}

MetricMode Metric::Mode() const {
    return mode_;
}

double Metric::Alpha() const {
    return alpha_;
}

double Metric::Beta() const {
    return beta_;
}

bool Metric::IsSymmetric() const {
    return kind_ != MetricKind::Tversky || alpha_ == beta_;
}

void Metric::ValidateForPDist() const {
    if (!IsSymmetric()) {
        throw std::invalid_argument("Asymmetric metrics are not valid for pairwise distances.");
    }
}

Metric::Metric(MetricKind kind, MetricMode mode, double alpha, double beta)
    : kind_(kind),
      mode_(mode),
      alpha_(alpha),
      beta_(beta) {
}

} // namespace OEFP
