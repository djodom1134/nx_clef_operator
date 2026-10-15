#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "config.h"

namespace clef_operator {

enum class MotionRank { onset = 0, active = 1, still = 2 };

const char* toString(MotionRank rank);

/** Motion inside one window, from the Server's per-frame motion grids (mirrors clef_operator/motion.py). */
struct MotionSummary
{
    int frames = 0;
    int framesWithMotion = 0;
    std::array<uint16_t, kGridW * kGridH> heat{};
    bool onset = false;

    /** Share of frames with motion; with ``only``, counts the busiest cell inside that mask. */
    double share(const CellMask* only = nullptr) const;
    MotionRank rank(double activeShare, const CellMask* only = nullptr) const;
};

struct Window
{
    int64_t startUs = 0;
    int64_t endUs = 0;
    std::string codec; //< "h264" or "hevc"
    std::string annexB; //< Elementary stream starting at a keyframe.
    int frames = 0;
    double fps = 0;
    MotionSummary motion;
    std::vector<int64_t> frameTimestampsUs; //< Every video frame in the window.
    std::vector<std::pair<int64_t, CellMask>> motionFrames; //< Frames whose grid had motion.
};

/** A frame and box to show as an event's thumbnail (normalized coordinates). */
struct BestShot
{
    int64_t timestampUs = 0;
    double x = 0, y = 0, w = 1, h = 1;
    bool fromMotion = false;
};

/**
 * The frame with the most motion inside ``zone`` (whole frame when null), boxed around that motion with one cell
 * of margin. Without motion: the middle frame of the window, boxed to the zone or the whole frame.
 */
BestShot bestShot(const Window& window, const CellMask* zone, const std::array<double, 4>* zoneBox);

/** Frames in the window with any motion inside ``zone`` (any motion at all when null). */
int framesWithMotionIn(const Window& window, const CellMask* zone);

/** Longest stretch of consecutive frames with motion inside ``zone``, in microseconds. */
int64_t longestMotionRunUs(const Window& window, const CellMask* zone);

/**
 * Motion rank for questions on ``zone`` (the whole frame when null). Counts frames with any motion in the zone, so a
 * car that crosses in two seconds counts even though each cell it passes moves for only a few frames. Active when
 * that is at least ``activeShare`` of the frames, or when the motion is continuous for at least half a second (a
 * passing car), which scattered noise is not.
 */
MotionRank zoneMotionRank(const Window& window, const CellMask* zone, double activeShare);

/** Ring buffer of compressed video packets and motion frames for one camera. */
class ClipBuffer
{
public:
    void addVideo(int64_t timestampUs, bool keyFrame, const std::string& codec, std::string data,
        const std::string& extradata);
    void addMotion(int64_t timestampUs, const CellMask& cells, bool any);

    /** Drop everything older than ``keepFromUs``, keeping the keyframe that starts the oldest window. */
    void trim(int64_t keepFromUs);

    /** Cut [endUs - lengthUs, endUs], with the start moved back to the nearest keyframe. */
    bool cut(int64_t endUs, int64_t lengthUs, Window* out) const;

    int64_t lastVideoUs() const { return m_video.empty() ? 0 : m_video.back().timestampUs; }

private:
    struct Packet
    {
        int64_t timestampUs;
        bool keyFrame;
        std::string data;
    };
    struct MotionFrame
    {
        int64_t timestampUs;
        CellMask cells;
        bool any;
    };

    std::deque<Packet> m_video;
    std::deque<MotionFrame> m_motion;
    std::string m_codec;
    std::string m_extradata; //< Annex B parameter sets, if the stream supplies them out of band.
};

/** Convert length-prefixed (AVCC/HVCC) NAL units to Annex B; Annex B input is returned unchanged. */
std::string toAnnexB(const std::string& data);

} // namespace clef_operator
