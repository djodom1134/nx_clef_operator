#include "clip_buffer.h"

#include <algorithm>

namespace clef_operator {

const char* toString(MotionRank rank)
{
    switch (rank)
    {
        case MotionRank::onset: return "onset";
        case MotionRank::active: return "active";
        default: return "still";
    }
}

double MotionSummary::share(const CellMask* only) const
{
    if (frames == 0)
        return 0;
    if (!only)
        return (double) framesWithMotion / frames;
    int busiest = 0;
    for (size_t i = 0; i < heat.size(); ++i)
    {
        if ((*only)[i])
            busiest = std::max(busiest, (int) heat[i]);
    }
    return (double) busiest / frames;
}

MotionRank MotionSummary::rank(double activeShare, const CellMask* only) const
{
    if (share(only) < activeShare)
        return MotionRank::still;
    return onset ? MotionRank::onset : MotionRank::active;
}

static bool startsWithStartCode(const std::string& d)
{
    return (d.size() >= 4 && d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 1)
        || (d.size() >= 3 && d[0] == 0 && d[1] == 0 && d[2] == 1);
}

std::string toAnnexB(const std::string& data)
{
    if (data.empty() || startsWithStartCode(data))
        return data;
    std::string out;
    out.reserve(data.size() + 16);
    size_t pos = 0;
    while (pos + 4 <= data.size())
    {
        const uint32_t length = ((uint8_t) data[pos] << 24) | ((uint8_t) data[pos + 1] << 16)
            | ((uint8_t) data[pos + 2] << 8) | (uint8_t) data[pos + 3];
        pos += 4;
        if (length == 0 || pos + length > data.size())
            return data; //< Not length-prefixed after all; pass through.
        out.append("\0\0\0\1", 4);
        out.append(data, pos, length);
        pos += length;
    }
    return out;
}

void ClipBuffer::addVideo(int64_t timestampUs, bool keyFrame, const std::string& codec, std::string data,
    const std::string& extradata)
{
    m_codec = codec;
    if (!extradata.empty() && startsWithStartCode(extradata))
        m_extradata = extradata;
    m_video.push_back({timestampUs, keyFrame, toAnnexB(data)});
}

void ClipBuffer::addMotion(int64_t timestampUs, const CellMask& cells, bool any)
{
    m_motion.push_back({timestampUs, cells, any});
}

void ClipBuffer::trim(int64_t keepFromUs)
{
    // Keep the last keyframe at or before keepFromUs, so a window starting there can still be cut.
    size_t firstKeep = 0;
    for (size_t i = 0; i < m_video.size() && m_video[i].timestampUs <= keepFromUs; ++i)
    {
        if (m_video[i].keyFrame)
            firstKeep = i;
    }
    m_video.erase(m_video.begin(), m_video.begin() + (long) firstKeep);
    while (!m_motion.empty() && m_motion.front().timestampUs < keepFromUs)
        m_motion.pop_front();
}

bool ClipBuffer::cut(int64_t endUs, int64_t lengthUs, Window* out) const
{
    const int64_t startUs = endUs - lengthUs;
    // Start at the last keyframe at or before startUs; if none, the first keyframe after it.
    long first = -1;
    for (size_t i = 0; i < m_video.size() && m_video[i].timestampUs <= startUs; ++i)
    {
        if (m_video[i].keyFrame)
            first = (long) i;
    }
    if (first < 0)
    {
        for (size_t i = 0; i < m_video.size(); ++i)
        {
            if (m_video[i].keyFrame && m_video[i].timestampUs < endUs)
            {
                first = (long) i;
                break;
            }
        }
    }
    if (first < 0)
        return false;

    Window w;
    w.codec = m_codec;
    w.startUs = m_video[(size_t) first].timestampUs;
    w.endUs = endUs;
    w.annexB = m_extradata;
    for (size_t i = (size_t) first; i < m_video.size() && m_video[i].timestampUs <= endUs; ++i)
    {
        w.annexB += m_video[i].data;
        w.frameTimestampsUs.push_back(m_video[i].timestampUs);
        ++w.frames;
    }
    const double seconds = (endUs - w.startUs) / 1e6;
    if (w.frames < 2 || seconds <= 0)
        return false;
    w.fps = w.frames / seconds;

    for (const auto& m: m_motion)
    {
        if (m.timestampUs < w.startUs || m.timestampUs > endUs)
            continue;
        ++w.motion.frames;
        if (!m.any)
            continue;
        ++w.motion.framesWithMotion;
        w.motionFrames.push_back({m.timestampUs, m.cells});
        for (size_t c = 0; c < m.cells.size(); ++c)
        {
            if (m.cells[c])
                ++w.motion.heat[c];
        }
    }
    *out = std::move(w);
    return true;
}

BestShot bestShot(const Window& window, const CellMask* zone, const std::array<double, 4>* zoneBox)
{
    BestShot shot;
    int bestCount = 0;
    const CellMask* bestCells = nullptr;
    for (const auto& [ts, cells]: window.motionFrames)
    {
        int count = 0;
        for (size_t c = 0; c < cells.size(); ++c)
            count += cells[c] && (!zone || (*zone)[c]);
        if (count > bestCount)
        {
            bestCount = count;
            bestCells = &cells;
            shot.timestampUs = ts;
        }
    }
    if (bestCells)
    {
        int x0 = kGridW, y0 = kGridH, x1 = -1, y1 = -1;
        for (int y = 0; y < kGridH; ++y)
        {
            for (int x = 0; x < kGridW; ++x)
            {
                const size_t c = (size_t) (y * kGridW + x);
                if ((*bestCells)[c] && (!zone || (*zone)[c]))
                {
                    x0 = std::min(x0, x); y0 = std::min(y0, y);
                    x1 = std::max(x1, x); y1 = std::max(y1, y);
                }
            }
        }
        x0 = std::max(0, x0 - 1); y0 = std::max(0, y0 - 1);
        x1 = std::min(kGridW - 1, x1 + 1); y1 = std::min(kGridH - 1, y1 + 1);
        shot.x = (double) x0 / kGridW;
        shot.y = (double) y0 / kGridH;
        shot.w = (double) (x1 - x0 + 1) / kGridW;
        shot.h = (double) (y1 - y0 + 1) / kGridH;
        shot.fromMotion = true;
        return shot;
    }
    if (!window.frameTimestampsUs.empty())
        shot.timestampUs = window.frameTimestampsUs[window.frameTimestampsUs.size() / 2];
    else
        shot.timestampUs = window.endUs;
    if (zoneBox)
    {
        shot.x = (*zoneBox)[0]; shot.y = (*zoneBox)[1];
        shot.w = (*zoneBox)[2]; shot.h = (*zoneBox)[3];
    }
    return shot;
}

static bool movesIn(const CellMask& cells, const CellMask* zone)
{
    for (size_t c = 0; c < cells.size(); ++c)
    {
        if (cells[c] && (!zone || (*zone)[c]))
            return true;
    }
    return false;
}

int framesWithMotionIn(const Window& window, const CellMask* zone)
{
    if (!zone)
        return window.motion.framesWithMotion;
    int frames = 0;
    for (const auto& [ts, cells]: window.motionFrames)
        frames += movesIn(cells, zone);
    return frames;
}

int64_t longestMotionRunUs(const Window& window, const CellMask* zone)
{
    // Frames with motion in the zone that follow each other with no motionless frame between them (the motion grid
    // comes with every video frame, so a gap of more than about 1.5 frame intervals means a still frame).
    const int64_t frameUs = window.fps > 0 ? (int64_t) (1e6 / window.fps) : 200'000;
    int64_t longest = 0, runStart = -1, previous = -1;
    for (const auto& [ts, cells]: window.motionFrames)
    {
        if (!movesIn(cells, zone))
            continue;
        if (previous < 0 || ts - previous > frameUs * 3 / 2)
            runStart = ts;
        previous = ts;
        longest = std::max(longest, ts - runStart + frameUs);
    }
    return longest;
}

MotionRank zoneMotionRank(const Window& window, const CellMask* zone, double activeShare)
{
    const int grids = window.motion.frames;
    if (grids == 0)
        return MotionRank::still;
    // Active with enough motion overall, or with half a second of continuous motion: a car crossing the frame in a
    // couple of seconds is continuous but small, while sensor noise and flicker are scattered single frames.
    const bool active = (double) framesWithMotionIn(window, zone) / grids >= activeShare
        || longestMotionRunUs(window, zone) >= 500'000;
    if (!active)
        return MotionRank::still;
    return window.motion.onset ? MotionRank::onset : MotionRank::active;
}

} // namespace clef_operator
