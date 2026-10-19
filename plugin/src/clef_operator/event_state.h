#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "config.h"

namespace clef_operator {

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
