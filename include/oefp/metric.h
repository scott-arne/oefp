#ifndef OEFP_METRIC_H
#define OEFP_METRIC_H

namespace OEFP {

/// \brief Fingerprint comparison metric family.
enum class MetricKind {
    Tanimoto,
    Jaccard,
    Tversky,
    Dice,
    Cosine,
    Manhattan,
};

/// \brief Whether a metric returns similarity or distance values.
enum class MetricMode {
    Similarity,
    Distance,
};

/// \brief Value-semantic metric configuration for fingerprint comparisons.
class Metric {
public:
    /// \brief Create a Tanimoto metric.
    static Metric Tanimoto(MetricMode mode = MetricMode::Similarity);

    /// \brief Create a Jaccard metric.
    static Metric Jaccard(MetricMode mode = MetricMode::Similarity);

    /// \brief Create a Tversky metric.
    ///
    /// \param alpha Weight for bits present only in the first fingerprint.
    /// \param beta Weight for bits present only in the second fingerprint.
    /// \param mode Whether to return similarity or distance values.
    /// \raises std::invalid_argument: When alpha or beta is outside [0.0, 1.0].
    static Metric Tversky(
        double alpha,
        double beta,
        MetricMode mode = MetricMode::Similarity);

    /// \brief Create a Dice metric.
    static Metric Dice(MetricMode mode = MetricMode::Similarity);

    /// \brief Create a Cosine metric.
    static Metric Cosine(MetricMode mode = MetricMode::Similarity);

    /// \brief Create a Manhattan distance metric.
    ///
    /// \raises std::invalid_argument: When mode is MetricMode::Similarity.
    static Metric Manhattan(MetricMode mode = MetricMode::Distance);

    /// \brief Return the metric family.
    MetricKind Kind() const;

    /// \brief Return whether this metric returns similarity or distance values.
    MetricMode Mode() const;

    /// \brief Return the Tversky alpha parameter.
    double Alpha() const;

    /// \brief Return the Tversky beta parameter.
    double Beta() const;

    /// \brief Return whether this metric is symmetric in its two inputs.
    bool IsSymmetric() const;

    /// \brief Validate that this metric is usable for pairwise distances.
    ///
    /// \raises std::invalid_argument: When the metric is asymmetric.
    void ValidateForPDist() const;

private:
    Metric(MetricKind kind, MetricMode mode, double alpha, double beta);

    MetricKind kind_;
    MetricMode mode_;
    double alpha_;
    double beta_;
};

} // namespace OEFP

#endif // OEFP_METRIC_H
