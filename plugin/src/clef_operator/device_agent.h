#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nx/sdk/analytics/helpers/consuming_device_agent.h>
#include <nx/sdk/analytics/i_motion_metadata_packet.h>

#include "clip_buffer.h"
#include "config.h"
#include "event_state.h"

namespace clef_operator {

class Engine;

/**
 * One per camera. Buffers the stream and the Server's motion grid, closes a 30 s window every 20 s (plus an early
 * window after motion onset), sends one gateway job per (fps, priority, zone) group, and turns answers into events.
 */
class DeviceAgent: public nx::sdk::analytics::ConsumingDeviceAgent
{
public:
    DeviceAgent(Engine* engine, const nx::sdk::IDeviceInfo* deviceInfo);
    virtual ~DeviceAgent() override;

protected:
    virtual std::string manifestString() const override;
    virtual nx::sdk::Result<const nx::sdk::ISettingsResponse*> settingsReceived() override;
    virtual void doPushDataPacket(nx::sdk::Result<void>* outResult, nx::sdk::analytics::IDataPacket* dataPacket) override;
    virtual bool pushCompressedVideoFrame(
        nx::sdk::Ptr<const nx::sdk::analytics::ICompressedVideoPacket> videoFrame) override;

private:
    void pushMotion(const nx::sdk::analytics::IMotionMetadataPacket* motion);
    /** PTZ hunter cameras: the motion grid, OR-ed over half a second, to the gateway (POST /v1/motion). */
    void streamHunterMotion(int64_t timestampUs, const CellMask& cells);
    /** Start, update or stop this camera's PTZ hunter in the gateway; returns a status line for the page. */
    std::string updateHunter(const CameraConfig& camera, const SystemConfig& system);
    void closeWindow(int64_t endUs, bool onsetWindow);
    void processWindow(Window window);
    void emitEvent(const QuestionConfig& q, const std::string& priority, const Zone* zone, const Transition& t,
        const Window& window, double fps);
    struct OpenEvent
    {
        std::string typeId;
        std::string caption;
        std::string key;
        nx::sdk::Uuid trackId; //< The object track that carries the event's thumbnail.
    };
    void emitClosed(const OpenEvent& open, const std::string& reason);
    void emitHealth(const std::string& caption, const std::string& description);

    void runTasks();
    void post(std::function<void()> task);

    Engine* const m_engine;
    const std::string m_cameraId;
    const std::string m_cameraName;

    // Push thread only.
    ClipBuffer m_buffer;
    int64_t m_nextWindowEndUs = 0;
    int64_t m_onsetWindowEndUs = 0;
    int64_t m_lastMotionUs = 0;
    int64_t m_lastOnsetUs = 0;
    std::atomic<int64_t> m_lastFrameUs{0};

    // Guarded by m_mutex.
    std::mutex m_mutex;
    CameraConfig m_camera;
    std::map<std::string, EventState> m_events; //< by question id
    std::set<std::string> m_liveQuestions; //< ids of the questions currently configured
    std::map<std::string, OpenEvent> m_openEvents; //< by question id, while its event is open
    int64_t m_lastHealthUs = 0;
    bool m_hunterWasEnabled = false;
    std::atomic<bool> m_hunterOn{false};
    // Push thread only: the hunter's motion stream.
    CellMask m_hunterCells{};
    int64_t m_hunterSentUs = 0;
    std::shared_ptr<std::atomic<bool>> m_motionInFlight = std::make_shared<std::atomic<bool>>(false);

    // Worker pool for gateway requests (they block for seconds).
    std::mutex m_taskMutex;
    std::condition_variable m_taskWake;
    std::deque<std::function<void()>> m_tasks;
    bool m_stopping = false;
    std::vector<std::thread> m_workers;
};

} // namespace clef_operator
