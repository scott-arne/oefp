#ifndef OEFP_METRIC_H
#define OEFP_METRIC_H

#include <vector>

namespace OEFP {

/// \brief Fingerprint comparison metric formula.
enum class MetricName {
    Euclidean,
    Manhattan,
    Chebyshev,
    Minkowski,
    StandardizedEuclidean,
    Mahalanobis,
    Haversine,
    Hamming,
    Canberra,
    BrayCurtis,
    Jaccard,
    Matching,
    Dice,
    Kulsinski,
    RogersTanimoto,
    RussellRao,
    SokalMichener,
    SokalSneath,
    Tanimoto,
    Tversky,
};

/// \brief Whether a metric returns distance or similarity values.
enum class MetricType {
    Distance,
    Similarity,
};

/// \brief Vector space the metric is intended to compare.
enum class MetricSpace {
    Real,
    Integer,
    Boolean,
};

/// \brief Value-semantic metric configuration for fingerprint comparisons.
class Metric {
public:
    /// \brief Create a Euclidean distance metric.
    static Metric Euclidean();

    /// \brief Create a Manhattan distance metric.
    static Metric Manhattan();

    /// \brief Create a Chebyshev distance metric.
    static Metric Chebyshev();

    /// \brief Create an unweighted Minkowski distance metric.
    ///
    /// \param p Minkowski exponent.
    /// \throws std::invalid_argument: When p is not greater than zero.
    static Metric Minkowski(double p = 2.0);

    /// \brief Create a weighted Minkowski distance metric.
    ///
    /// \param p Minkowski exponent.
    /// \param weights Per-dimension non-negative weights.
    /// \throws std::invalid_argument: When p is not greater than zero or any
    ///     weight is negative.
    static Metric Minkowski(double p, std::vector<double> weights);

    /// \brief Create a standardized Euclidean distance metric.
    ///
    /// \param variances Per-dimension variance values.
    static Metric StandardizedEuclidean(std::vector<double> variances);

    /// \brief Create a Mahalanobis distance metric.
    ///
    /// An asymmetric matrix is interpreted as its symmetric part, (VI + VI^T) / 2, which is the
    /// value the quadratic form d^T * VI * d defines.
    ///
    /// \param inverse_covariance Row-major inverse covariance matrix.
    static Metric Mahalanobis(std::vector<double> inverse_covariance);

    /// \brief Create a Haversine distance metric.
    static Metric Haversine();

    /// \brief Create a Hamming distance metric.
    static Metric Hamming();

    /// \brief Create a Canberra distance metric.
    static Metric Canberra();

    /// \brief Create a Bray-Curtis distance metric.
    ///
    /// Computed as sum|a-b| / (sum|a| + sum|b|), which is bounded in [0, 1] for any
    /// real input. The classical ecology form sum|a-b| / sum|a+b| (used by scipy)
    /// agrees with this for non-negative input, but is unbounded and undefined at
    /// a = -b, so it is unsuitable for signed descriptors such as partial charges or
    /// BCUT2D eigenvalues.
    /// When both inputs have zero absolute-value mass the quotient is undefined; OEFP
    /// returns 0.0, where scipy returns NaN.
    static Metric BrayCurtis();

    /// \brief Create a Jaccard distance metric.
    static Metric Jaccard();

    /// \brief Create a Matching distance metric.
    static Metric Matching();

    /// \brief Create a Dice distance metric.
    ///
    /// When both inputs are empty the quotient is undefined; OEFP returns 0.0, where scipy
    /// returns NaN. Note that Dice does not satisfy the triangle inequality -- see
    /// SatisfiesTriangleInequality().
    static Metric Dice();

    /// \brief Create a Kulsinski distance metric.
    static Metric Kulsinski();

    /// \brief Create a Rogers-Tanimoto distance metric.
    static Metric RogersTanimoto();

    /// \brief Create a Russell-Rao distance metric.
    static Metric RussellRao();

    /// \brief Create a Sokal-Michener distance metric.
    static Metric SokalMichener();

    /// \brief Create a Sokal-Sneath distance metric.
    ///
    /// When both inputs are empty the quotient is undefined; OEFP returns 0.0, where scipy
    /// raises.
    static Metric SokalSneath();

    /// \brief Create a Tanimoto similarity metric.
    static Metric Tanimoto();

    /// \brief Create a Tversky similarity metric.
    ///
    /// \param alpha Weight for dimensions present only in the first fingerprint.
    /// \param beta Weight for dimensions present only in the second fingerprint.
    /// \throws std::invalid_argument: When alpha or beta is outside [0.0, 1.0].
    static Metric Tversky(double alpha, double beta);

    /// \brief Return the metric formula.
    MetricName Name() const;

    /// \brief Return whether this metric is a distance or similarity.
    MetricType Type() const;

    /// \brief Return the vector space this metric is intended for.
    MetricSpace Space() const;

    /// \brief Return the Minkowski exponent.
    double P() const;

    /// \brief Return the Tversky alpha parameter.
    double Alpha() const;

    /// \brief Return the Tversky beta parameter.
    double Beta() const;

    /// \brief Return weighted Minkowski weights.
    const std::vector<double>& Weights() const;

    /// \brief Return standardized Euclidean variances.
    const std::vector<double>& Variances() const;

    /// \brief Return Mahalanobis inverse covariance values.
    const std::vector<double>& InverseCovariance() const;

    /// \brief Return whether this metric is symmetric in its two inputs.
    bool IsSymmetric() const;

    /// \brief Return whether comparing a value with itself yields exactly zero.
    ///
    /// False for Kulsinski and Russell-Rao, whose self-distance is the fraction of
    /// dimensions that are zero in both inputs, and for the similarity metrics, whose
    /// self-comparison is 1.0 rather than 0.0. Callers that treat similarities and
    /// distances differently should consult Type() first: this predicate answers one
    /// question, and answers it about the returned number, not about the metric's intent.
    ///
    /// Metrics whose behaviour depends on their parameters report the property that holds
    /// when those parameters are valid. Standardized Euclidean assumes finite, strictly
    /// positive variances; Mahalanobis assumes a symmetric positive semidefinite inverse
    /// covariance; Haversine assumes coordinates are radian latitude and longitude within
    /// their valid ranges. Those preconditions are not checked at construction.
    bool HasZeroSelfDistance() const;

    /// \brief Return whether this metric satisfies the triangle inequality.
    ///
    /// False for Dice and Bray-Curtis, for Minkowski with an exponent below 1.0, and for
    /// the similarity metrics. Dice's violation is exhibited by nested sets at any
    /// dimensionality: for A = {0}, B = {0, 1}, C = {1}, d(A, B) + d(B, C) is 2/3 while
    /// d(A, C) is 1.
    ///
    /// The parameter-dependent metrics carry the same preconditions documented on
    /// HasZeroSelfDistance().
    bool SatisfiesTriangleInequality() const;

    /// \brief Validate that this metric is usable for pairwise comparisons.
    ///
    /// \throws std::invalid_argument: When the metric is asymmetric.
    void ValidateForPDist() const;

    /// \brief Validate that this metric is a true distance metric.
    ///
    /// Checks that the metric returns distances rather than similarities, is symmetric,
    /// has zero self-distance, and satisfies the triangle inequality. Algorithms that
    /// require a metric space -- ball trees, metric indexes, and clustering methods that
    /// assume the triangle inequality -- can gate on this instead of maintaining their own
    /// list of which metrics qualify.
    ///
    /// \throws std::invalid_argument: When the metric fails any of those properties.
    void ValidateAsDistanceMetric() const;

private:
    Metric(
        MetricName name,
        MetricType type,
        MetricSpace space,
        double p,
        double alpha,
        double beta,
        std::vector<double> weights,
        std::vector<double> variances,
        std::vector<double> inverse_covariance);

    MetricName name_;
    MetricType type_;
    MetricSpace space_;
    double p_;
    double alpha_;
    double beta_;
    std::vector<double> weights_;
    std::vector<double> variances_;
    std::vector<double> inverse_covariance_;
};

} // namespace OEFP

#endif // OEFP_METRIC_H
