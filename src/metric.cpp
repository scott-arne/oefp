#include "oefp/metric.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace OEFP {
namespace {

void validate_positive(double value, const char* name) {
    if (!(value > 0.0)) {
        throw std::invalid_argument(std::string(name) + " must be greater than zero.");
    }
}

void validate_non_negative_weights(const std::vector<double>& weights) {
    for (const auto weight : weights) {
        if (!(weight >= 0.0)) {
            throw std::invalid_argument("Minkowski weights cannot be negative.");
        }
    }
}

void validate_tversky_parameter(double value, const char* name) {
    if (!(value >= 0.0 && value <= 1.0)) {
        throw std::invalid_argument(std::string("Tversky ") + name + " must be in [0.0, 1.0].");
    }
}

} // namespace

Metric Metric::Euclidean() {
    return Metric(MetricName::Euclidean, MetricType::Distance, MetricSpace::Real, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Manhattan() {
    return Metric(MetricName::Manhattan, MetricType::Distance, MetricSpace::Real, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Chebyshev() {
    return Metric(MetricName::Chebyshev, MetricType::Distance, MetricSpace::Real, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Minkowski(double p) {
    validate_positive(p, "Minkowski p");
    return Metric(MetricName::Minkowski, MetricType::Distance, MetricSpace::Real, p, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Minkowski(double p, std::vector<double> weights) {
    validate_positive(p, "Minkowski p");
    validate_non_negative_weights(weights);
    return Metric(
        MetricName::Minkowski,
        MetricType::Distance,
        MetricSpace::Real,
        p,
        1.0,
        1.0,
        std::move(weights),
        {},
        {});
}

Metric Metric::StandardizedEuclidean(std::vector<double> variances) {
    return Metric(
        MetricName::StandardizedEuclidean,
        MetricType::Distance,
        MetricSpace::Real,
        2.0,
        1.0,
        1.0,
        {},
        std::move(variances),
        {});
}

Metric Metric::Mahalanobis(std::vector<double> inverse_covariance) {
    return Metric(
        MetricName::Mahalanobis,
        MetricType::Distance,
        MetricSpace::Real,
        2.0,
        1.0,
        1.0,
        {},
        {},
        std::move(inverse_covariance));
}

Metric Metric::Haversine() {
    return Metric(MetricName::Haversine, MetricType::Distance, MetricSpace::Real, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Hamming() {
    return Metric(MetricName::Hamming, MetricType::Distance, MetricSpace::Integer, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Canberra() {
    return Metric(MetricName::Canberra, MetricType::Distance, MetricSpace::Integer, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::BrayCurtis() {
    return Metric(MetricName::BrayCurtis, MetricType::Distance, MetricSpace::Integer, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Jaccard() {
    return Metric(MetricName::Jaccard, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Matching() {
    return Metric(MetricName::Matching, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Dice() {
    return Metric(MetricName::Dice, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Kulsinski() {
    return Metric(MetricName::Kulsinski, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::RogersTanimoto() {
    return Metric(MetricName::RogersTanimoto, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::RussellRao() {
    return Metric(MetricName::RussellRao, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::SokalMichener() {
    return Metric(MetricName::SokalMichener, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::SokalSneath() {
    return Metric(MetricName::SokalSneath, MetricType::Distance, MetricSpace::Boolean, 2.0, 1.0, 1.0, {}, {}, {});
}

Metric Metric::Tanimoto() {
    return Metric(
        MetricName::Tanimoto,
        MetricType::Similarity,
        MetricSpace::Boolean,
        2.0,
        1.0,
        1.0,
        {},
        {},
        {});
}

Metric Metric::Tversky(double alpha, double beta) {
    validate_tversky_parameter(alpha, "alpha");
    validate_tversky_parameter(beta, "beta");
    return Metric(
        MetricName::Tversky,
        MetricType::Similarity,
        MetricSpace::Boolean,
        2.0,
        alpha,
        beta,
        {},
        {},
        {});
}

MetricName Metric::Name() const {
    return name_;
}

MetricType Metric::Type() const {
    return type_;
}

MetricSpace Metric::Space() const {
    return space_;
}

double Metric::P() const {
    return p_;
}

double Metric::Alpha() const {
    return alpha_;
}

double Metric::Beta() const {
    return beta_;
}

const std::vector<double>& Metric::Weights() const {
    return weights_;
}

const std::vector<double>& Metric::Variances() const {
    return variances_;
}

const std::vector<double>& Metric::InverseCovariance() const {
    return inverse_covariance_;
}

bool Metric::IsSymmetric() const {
    return name_ != MetricName::Tversky || alpha_ == beta_;
}

bool Metric::HasZeroSelfDistance() const {
    // Switching without a default makes a new MetricName a compile error here rather than a
    // silently wrong capability answer.
    switch (name_) {
    case MetricName::Kulsinski:
    case MetricName::RussellRao:
    case MetricName::Tanimoto:
    case MetricName::Tversky:
        return false;
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
    case MetricName::Haversine:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
    case MetricName::Jaccard:
    case MetricName::Matching:
    case MetricName::Dice:
    case MetricName::RogersTanimoto:
    case MetricName::SokalMichener:
    case MetricName::SokalSneath:
        return true;
    }

    return false;
}

bool Metric::SatisfiesTriangleInequality() const {
    switch (name_) {
    case MetricName::Minkowski:
        return p_ >= 1.0;
    case MetricName::Dice:
    case MetricName::BrayCurtis:
    case MetricName::Tanimoto:
    case MetricName::Tversky:
        return false;
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
    case MetricName::Haversine:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::Jaccard:
    case MetricName::Matching:
    case MetricName::Kulsinski:
    case MetricName::RogersTanimoto:
    case MetricName::RussellRao:
    case MetricName::SokalMichener:
    case MetricName::SokalSneath:
        return true;
    }

    return false;
}

void Metric::ValidateForPDist() const {
    if (!IsSymmetric()) {
        throw std::invalid_argument("Asymmetric metrics are not valid for pairwise distances.");
    }
}

void Metric::ValidateAsDistanceMetric() const {
    if (type_ != MetricType::Distance) {
        throw std::invalid_argument("Similarity metrics are not valid distance metrics.");
    }
    ValidateForPDist();
    if (!HasZeroSelfDistance()) {
        throw std::invalid_argument(
            "Metrics without zero self-distance are not valid distance metrics.");
    }
    if (!SatisfiesTriangleInequality()) {
        throw std::invalid_argument(
            "Metrics that violate the triangle inequality are not valid distance metrics.");
    }
}

Metric::Metric(
    MetricName name,
    MetricType type,
    MetricSpace space,
    double p,
    double alpha,
    double beta,
    std::vector<double> weights,
    std::vector<double> variances,
    std::vector<double> inverse_covariance)
    : name_(name),
      type_(type),
      space_(space),
      p_(p),
      alpha_(alpha),
      beta_(beta),
      weights_(std::move(weights)),
      variances_(std::move(variances)),
      inverse_covariance_(std::move(inverse_covariance)) {
}

} // namespace OEFP
