#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>

#include "config.h"

namespace clef_operator {

/**
 * Short text for the answers shown on the video: a true/false question shows "YES 0.92" at or above its
 * on-threshold and "no 0.02" below it; choice and scale questions show their most likely option ("closed 0.85"),
 * in capitals when the answer is alerting.
 */
inline std::string answerText(
    const QuestionConfig& q, const std::map<std::string, double>& answer, double alertP)
{
    char number[16];
    const bool alert = alertP >= q.onThreshold;
    if (q.kind == "noul")
    {
        std::snprintf(number, sizeof(number), "%.2f", alertP);
        return std::string(alert ? "YES " : "no ") + number;
    }
    std::string best;
    double bestP = -1;
    for (const auto& [option, p]: answer)
    {
        if (p > bestP)
            best = option, bestP = p;
    }
    if (best.empty())
        return "-";
    if (alert)
        std::transform(best.begin(), best.end(), best.begin(), [](unsigned char c) { return (char) std::toupper(c); });
    std::snprintf(number, sizeof(number), "%.2f", bestP);
    return best + " " + number;
}

/** The label for a question in the answers shown on the video: its text, shortened, after its zone if any. */
inline std::string answerLabel(const std::string& text, const std::string& zone, size_t maxLength = 40)
{
    std::string label = zone.empty() ? text : zone + ": " + text;
    if (label.size() > maxLength)
        label = label.substr(0, maxLength - 3) + "...";
    return label;
}

/** Alert probability of one answer ({option: probability}), as in clef_operator/events.py. */
inline double alertProbability(const QuestionConfig& q, const std::map<std::string, double>& answer)
{
    const auto value =
        [&](const std::string& key)
        {
            const auto it = answer.find(key);
            return it == answer.end() ? 0.0 : it->second;
        };
    if (q.kind == "noul")
        return value("true");
    if (q.kind == "choice")
    {
        double total = 0;
        for (const auto& option: q.alertOn)
            total += value(option);
        return total;
    }
    // score: probability at or above the lowest alerting level (answers keyed by level text or index).
    size_t lowest = q.options.empty() ? 0 : q.options.size() - 1;
    if (!q.alertOn.empty())
    {
        for (size_t i = 0; i < q.options.size(); ++i)
        {
            if (q.options[i] == q.alertOn[0])
                lowest = i;
        }
    }
    double total = 0;
    for (size_t i = lowest; i < q.options.size(); ++i)
        total += answer.count(q.options[i]) ? value(q.options[i]) : value(std::to_string(i));
    return total;
}

struct Transition
{
    bool open = false; //< false = close
    int64_t timeUs = 0;
    double probability = 0;
};

/** One per (camera, question): open at onThreshold, hold at offThreshold, close after N windows below. */
class EventState
{
public:
    std::optional<Transition> update(
        int64_t windowStartUs, int64_t windowEndUs, std::optional<double> probability,
        double onThreshold, double offThreshold, int closeAfter)
    {
        if (windowEndUs <= m_lastWindowEndUs)
            return std::nullopt; //< Windows are answered in parallel; ignore one older than the last applied.
        m_lastWindowEndUs = windowEndUs;
        const double p = probability.value_or(0.0);
        if (!m_active)
        {
            if (p < onThreshold)
                return std::nullopt;
            m_active = true;
            m_startedUs = windowStartUs;
            m_lastPositiveEndUs = windowEndUs;
            m_peak = p;
            m_below = 0;
            return Transition{true, windowStartUs, p};
        }
        if (p >= offThreshold)
        {
            m_below = 0;
            m_lastPositiveEndUs = windowEndUs;
            m_peak = std::max(m_peak, p);
            return std::nullopt;
        }
        if (++m_below < closeAfter)
            return std::nullopt;
        m_active = false;
        return Transition{false, m_lastPositiveEndUs, m_peak};
    }

    bool active() const { return m_active; }
    double peak() const { return m_peak; }

private:
    bool m_active = false;
    int64_t m_startedUs = 0;
    int64_t m_lastPositiveEndUs = 0;
    double m_peak = 0;
    int m_below = 0;
    int64_t m_lastWindowEndUs = 0;
};

} // namespace clef_operator
