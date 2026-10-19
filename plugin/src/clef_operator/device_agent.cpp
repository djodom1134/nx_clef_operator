#include "device_agent.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <set>

#include <nx/kit/debug.h>
#include <nx/kit/json.h>
#include <nx/sdk/analytics/helpers/event_metadata.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_metadata.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_track_best_shot_packet.h>
#include <nx/sdk/helpers/attribute.h>
#include <nx/sdk/helpers/uuid_helper.h>
#include <nx/sdk/analytics/i_compressed_video_packet.h>
#include <nx/sdk/helpers/settings_response.h>

#include "engine.h"
#include "http_client.h"

#undef NX_PRINT_PREFIX
#define NX_PRINT_PREFIX (this->logUtils.printPrefix)

namespace clef_operator {

using namespace nx::sdk;
using namespace nx::sdk::analytics;
using nx::kit::Json;

namespace {

constexpr int kWorkers = 3;
constexpr int64_t kSecondUs = 1'000'000;
constexpr int kGatewayTimeoutS = 180;
const std::string kAnswerObjectType = "nx.clef_operator.answer";

std::string eventTypeFor(const std::string& priority)
{
    if (priority == "critical" || priority == "high" || priority == "normal")
        return "nx.clef_operator." + priority;
    return "nx.clef_operator.low";
}

std::string clock(int64_t us)
{
    const time_t t = (time_t) (us / kSecondUs);
    tm local{};
    localtime_r(&t, &local);
    char text[16];
    strftime(text, sizeof(text), "%H:%M:%S", &local);
    return text;
}

std::string fixed(double v, int digits = 2)
{
    char text[32];
    snprintf(text, sizeof(text), "%.*f", digits, v);
    return text;
}

std::string fpsText(double fps)
{
    char text[16];
    snprintf(text, sizeof(text), "%g", fps);
    return text;
}

} // namespace

DeviceAgent::DeviceAgent(Engine* engine, const IDeviceInfo* deviceInfo):
    ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ false),
    m_engine(engine),
    m_cameraId(deviceInfo->id()),
    m_cameraName(deviceInfo->name())
{
    for (int i = 0; i < kWorkers; ++i)
        m_workers.emplace_back([this]() { runTasks(); });
}

DeviceAgent::~DeviceAgent()
{
    {
        std::lock_guard<std::mutex> lock(m_taskMutex);
        m_stopping = true;
        m_tasks.clear();
    }
    m_taskWake.notify_all();
    for (auto& worker: m_workers)
        worker.join();
}

std::string DeviceAgent::manifestString() const
{
    Json::array types;
    for (const char* id: {"nx.clef_operator.critical", "nx.clef_operator.high", "nx.clef_operator.normal",
        "nx.clef_operator.low", "nx.clef_operator.nuisance", "nx.clef_operator.health"})
    {
        types.push_back(Json::object{{"eventTypeId", id}});
    }
    types.push_back(Json::object{{"objectTypeId", kAnswerObjectType},
        {"attributes", Json::array{"Question", "Answer", "Probability", "Zone", "Priority"}}});
    return Json(Json::object{{"supportedTypes", types}}).dump();
}

Result<const ISettingsResponse*> DeviceAgent::settingsReceived()
{
    const auto system = m_engine->systemConfig();
    CameraConfig camera = parseCameraConfig(currentSettings(), *system);
    NX_PRINT << "Clef Operator: " << m_cameraName << ": " << camera.questions.size() << " questions, "
        << camera.zones.size() << " zones";
    for (const auto& q: camera.questions)
        m_engine->resolve(q, m_cameraName, camera.description);

    auto response = makePtr<SettingsResponse>();
    std::vector<std::string> zoneNames(kMaxZones);
    for (const auto& z: camera.zones)
        zoneNames[(size_t) z.index - 1] = z.name;
    // Always return the model: a response without one makes the Server treat the model as empty.
    response->setModel(deviceAgentSettingsModel(zoneNames)); //< Also shows zone names in each question's Zone list.
    for (const auto& q: camera.questions)
    {
        const std::string summary = m_engine->classificationSummary(q.text);
        response->setValue("q" + q.id.substr(1) + ".clef", summary.empty() ? "pending: reopen to see Clef's choice" : summary);
    }

    response->setValue("hunter.status", updateHunter(camera, *system));

    std::lock_guard<std::mutex> lock(m_mutex);
    std::set<std::string> live;
    for (const auto& q: camera.questions)
        live.insert(q.id);
    for (const auto& q: system->questions)
        live.insert(q.id);
    for (auto it = m_events.begin(); it != m_events.end();)
    {
        if (live.count(it->first))
        {
            ++it;
            continue;
        }
        const auto open = m_openEvents.find(it->first);
        if (it->second.active() && open != m_openEvents.end())
            emitClosed(open->second, "Question removed"); //< PRD: removing a question closes its event.
        m_openEvents.erase(it->first);
        it = m_events.erase(it);
    }
    m_liveQuestions = std::move(live);
    m_camera = std::move(camera);
    return response.releasePtr();
}

std::string DeviceAgent::updateHunter(const CameraConfig& camera, const SystemConfig& system)
{
    const HunterSettings& h = camera.hunter;
    Json::array magnets;
    for (const auto& m: h.magnets)
        magnets.push_back(Json::object{{"pan", m[0]}, {"tilt", m[1]}, {"fov", m[2]}});
    const Json body = Json::object{
        {"camera_id", m_cameraId}, {"enabled", h.enabled}, {"camera_name", m_cameraName},
        {"description", camera.description}, {"interests", system.siteInterests},
        {"targets", Json::array(h.targets.begin(), h.targets.end())},
        {"avoid", Json::array(h.avoid.begin(), h.avoid.end())}, {"magnets", magnets},
        {"cues", Json::array(h.cues.begin(), h.cues.end())},
        {"presets", Json::array(h.presets.begin(), h.presets.end())},
        {"threshold", h.threshold}, {"max_speed", h.maxSpeed}, {"rest_s", h.restSeconds},
        {"dwell_s", h.dwellSeconds}, {"pan_min", h.panMin}, {"pan_max", h.panMax},
        {"wide_share", h.widePercent / 100.0}, {"focus_s", h.focusSeconds}, {"handback_s", h.handBackSeconds},
    };
    if (!h.enabled && !m_hunterWasEnabled)
        return "off";
    m_hunterWasEnabled = h.enabled;
    const auto response = httpPostJson(system.gatewayUrl + "/v1/hunter", body.dump(), system.gatewayToken, 15);
    std::string error;
    const Json result = Json::parse(response.body, error);
    if (response.status != 200 || !error.empty())
        return "gateway unreachable: " + (response.error.empty() ? std::to_string(response.status) : response.error);
    const std::string status = result["status"].string_value();
    NX_PRINT << "Clef Operator: " << m_cameraName << ": PTZ hunter " << status << " " << result["error"].string_value();
    return status == "error" ? "not started: " + result["error"].string_value() : status;
}

void DeviceAgent::doPushDataPacket(Result<void>* outResult, IDataPacket* dataPacket)
{
    if (dataPacket)
    {
        if (const auto motion = dataPacket->queryInterface<IMotionMetadataPacket>())
        {
            pushMotion(motion.get());
            return;
        }
    }
    ConsumingDeviceAgent::doPushDataPacket(outResult, dataPacket);
}

void DeviceAgent::pushMotion(const IMotionMetadataPacket* motion)
{
    const int64_t ts = motion->timestampUs();
    CellMask cells{};
    bool any = false;
    if (!motion->isEmpty())
    {
        const int cols = motion->columnCount(), rows = motion->rowCount();
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                if (motion->isMotionAt(c, r))
                {
                    cells[(size_t) (r * kGridH / rows) * kGridW + (size_t) (c * kGridW / cols)] = true;
                    any = true;
                }
            }
        }
    }
    m_buffer.addMotion(ts, cells, any);
    if (!any)
        return;

    const auto system = m_engine->systemConfig();
    if (m_lastMotionUs == 0 || ts - m_lastMotionUs > system->quietSeconds * kSecondUs)
    {
        m_lastOnsetUs = ts;
        const int64_t candidate = ts + 10 * kSecondUs;
        if (system->onsetWindow && m_onsetWindowEndUs == 0 && m_nextWindowEndUs - candidate > 5 * kSecondUs)
            m_onsetWindowEndUs = candidate;
    }
    m_lastMotionUs = ts;
}

bool DeviceAgent::pushCompressedVideoFrame(Ptr<const ICompressedVideoPacket> frame)
{
    const int64_t ts = frame->timestampUs();
    m_lastFrameUs = ts;
    std::string extradata;
    if (const auto context = frame->context(); context && context->extradataSize() > 0)
        extradata.assign(context->extradata(), (size_t) context->extradataSize());
    const bool key = ((uint32_t) frame->flags() & (uint32_t) ICompressedMediaPacket::MediaFlags::keyFrame) != 0;
    m_buffer.addVideo(ts, key, frame->codec(), std::string(frame->data(), (size_t) frame->dataSize()), extradata);

    // The Server attaches its motion grid to the video frame (streamTypeFilter "motion").
    if (const auto metadata = frame->metadataList())
    {
        for (int i = 0; i < metadata->count(); ++i)
        {
            if (const auto packet = metadata->at(i))
            {
                if (const auto motion = packet->queryInterface<IMotionMetadataPacket>())
                    pushMotion(motion.get());
            }
        }
    }

    const auto system = m_engine->systemConfig();
    const int64_t windowUs = system->windowSeconds * kSecondUs;
    const int64_t stepUs = (system->windowSeconds - system->overlapSeconds) * kSecondUs;
    if (m_nextWindowEndUs == 0)
        m_nextWindowEndUs = ts + windowUs;

    if (m_onsetWindowEndUs != 0 && ts >= m_onsetWindowEndUs)
    {
        closeWindow(m_onsetWindowEndUs, /*onsetWindow*/ true);
        m_onsetWindowEndUs = 0;
    }
    if (ts >= m_nextWindowEndUs)
    {
        closeWindow(m_nextWindowEndUs, /*onsetWindow*/ false);
        while (m_nextWindowEndUs <= ts)
            m_nextWindowEndUs += stepUs;
    }
    m_buffer.trim(ts - windowUs - 10 * kSecondUs);
    return true;
}

void DeviceAgent::closeWindow(int64_t endUs, bool onsetWindow)
{
    const auto system = m_engine->systemConfig();
    Window window;
    if (!m_buffer.cut(endUs, system->windowSeconds * kSecondUs, &window))
        return;
    window.motion.onset = m_lastOnsetUs >= window.startUs && m_lastOnsetUs <= endUs;
    if (onsetWindow)
        NX_PRINT << "Clef Operator: " << m_cameraName << ": motion onset window";
    post([this, w = std::move(window)]() mutable { processWindow(std::move(w)); });
}

void DeviceAgent::processWindow(Window window)
{
    const auto system = m_engine->systemConfig();
    CameraConfig camera;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        camera = m_camera;
    }

    std::vector<QuestionConfig> questions;
    for (const auto& q: system->questions)
    {
        const int number = std::atoi(q.id.c_str() + 1);
        if (std::find(camera.systemQuestionsOff.begin(), camera.systemQuestionsOff.end(), number)
            == camera.systemQuestionsOff.end())
        {
            questions.push_back(q);
        }
    }
    questions.insert(questions.end(), camera.questions.begin(), camera.questions.end());
    if (questions.empty())
        return;

    const auto zoneFor =
        [&](int index) -> const Zone*
        {
            for (const auto& z: camera.zones)
            {
                if (z.index == index)
                    return &z;
            }
            return nullptr;
        };

    // Group questions that can share one Clef request: same fps, priority and zone.
    struct Group
    {
        double fps;
        std::string priority;
        const Zone* zone;
        MotionRank motion;
        std::vector<ResolvedQuestion> questions;
    };
    std::map<std::string, Group> groups;
    int skippedStill = 0;
    for (const auto& q: questions)
    {
        ResolvedQuestion r = m_engine->resolve(q, m_cameraName, camera.description);
        r.fps = std::max(1.0, std::min(r.fps, std::floor(window.fps))); //< Never ask for more frames than the stream has.
        const Zone* zone = zoneFor(q.zone);
        const MotionRank rank = zoneMotionRank(window, zone ? &zone->cells : nullptr, system->activeMotionShare);
        if (r.needsMotion && rank == MotionRank::still)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_liveQuestions.count(q.id))
                continue;
            const auto t = m_events[q.id].update(window.startUs, window.endUs, std::nullopt,
                q.onThreshold, q.offThreshold, system->closeAfter);
            if (t)
                emitEvent(q, r.priority, zone, *t, window, r.fps);
            ++skippedStill;
            continue;
        }
        const std::string key = fpsText(r.fps) + "|" + r.priority + "|" + std::to_string(q.zone);
        auto& g = groups.try_emplace(key, Group{r.fps, r.priority, zone, rank, {}}).first->second;
        g.questions.push_back(r);
    }
    std::string zoneSummary;
    for (const auto& z: camera.zones)
    {
        zoneSummary += "; " + z.name + " " + std::to_string(framesWithMotionIn(window, &z.cells)) + " frames "
            + toString(zoneMotionRank(window, &z.cells, system->activeMotionShare));
    }
    NX_PRINT << "Clef Operator: " << m_cameraName << ": window " << clock(window.startUs) << "-" << clock(window.endUs)
        << " " << window.frames << " frames (" << fixed(window.fps, 1) << " fps), motion "
        << fixed(window.motion.share() * 100, 0) << "% of " << window.motion.frames << " grids, "
        << toString(window.motion.rank(system->activeMotionShare)) << "; " << groups.size() << " requests, "
        << skippedStill << " questions skipped (no motion)" << zoneSummary;
    if (groups.empty())
        return;

    std::vector<std::string> nuisance = system->nuisanceSources;
    nuisance.insert(nuisance.end(), camera.extraNuisance.begin(), camera.extraNuisance.end());
    Json::array nuisanceJson(nuisance.begin(), nuisance.end());
    const std::string clip = base64Encode(window.annexB);

    // Send every group of this window at once, so the gateway's scheduler (not send order) decides what runs first:
    // a Normal question must not wait for this window's Critical answer before it even joins the queue.
    const auto submit =
        [&](Group& g)
        {
            Json::object questionsJson;
            Json::object alertsJson;
            for (const auto& r: g.questions)
            {
                const auto& q = r.config;
                Json::object qj{{"type", q.kind}, {"instructions", q.text}};
                if (q.kind == "choice")
                {
                    Json::object criteria;
                    for (const auto& o: q.options)
                        criteria[o] = o;
                    qj["criteria"] = criteria;
                }
                else if (q.kind == "score")
                {
                    qj["criteria"] = Json::array(q.options.begin(), q.options.end());
                }
                questionsJson[q.id] = qj;
                // For the gateway's dashboard: when this question's answer counts as yes.
                alertsJson[q.id] = Json::object{{"on", q.onThreshold}, {"alert_on", Json::array(q.alertOn.begin(), q.alertOn.end())}};
            }
            Json::object body{
                {"camera_id", m_cameraId}, {"camera_name", m_cameraName}, {"camera_description", camera.description},
                {"window_start", window.startUs / 1e6}, {"window_end", window.endUs / 1e6},
                {"priority", g.priority}, {"fps", g.fps}, {"motion", toString(g.motion)},
                {"site", Json::object{{"interests", system->siteInterests}, {"nuisance", nuisanceJson}}},
                {"questions", questionsJson}, {"alerts", alertsJson},
                {"clip", clip}, {"clip_format", "annexb"}, {"codec", window.codec}, {"clip_fps", window.fps},
                {"timeout", kGatewayTimeoutS},
            };
            if (g.zone)
            {
                body["zone"] = Json::object{{"name", g.zone->name}, {"description", g.zone->description},
                    {"box", Json::array{g.zone->box[0], g.zone->box[1], g.zone->box[2], g.zone->box[3]}}};
            }

            const auto response = httpPostJson(system->gatewayUrl + "/v1/jobs", Json(body).dump(),
                system->gatewayToken, kGatewayTimeoutS + 30);
            std::string error;
            const Json result = Json::parse(response.body, error);
            if (response.status != 200 || !error.empty())
            {
                NX_PRINT << "Clef Operator: " << m_cameraName << ": gateway error: "
                    << (response.error.empty() ? response.body.substr(0, 300) : response.error);
                emitHealth("Clef gateway unreachable", system->gatewayUrl + ": "
                    + (response.error.empty() ? "HTTP " + std::to_string(response.status) : response.error));
                return;
            }
            if (result["status"].string_value() != "answered")
            {
                NX_PRINT << "Clef Operator: " << m_cameraName << ": window " << clock(window.endUs) << " "
                    << g.priority << " " << result["status"].string_value() << " (" << result["reason"].string_value()
                    << result["error"].string_value() << ")";
                return;
            }

            std::lock_guard<std::mutex> lock(m_mutex);
            for (const auto& r: g.questions)
            {
                if (!m_liveQuestions.count(r.config.id))
                    continue; //< Removed or turned off while this window was in flight.
                std::map<std::string, double> answer;
                for (const auto& [option, p]: result["answers"][r.config.id].object_items())
                    answer[option] = p.number_value();
                const double p = alertProbability(r.config, answer);
                const auto t = m_events[r.config.id].update(window.startUs, window.endUs, p,
                    r.config.onThreshold, r.config.offThreshold, system->closeAfter);
                if (t)
                    emitEvent(r.config, g.priority, g.zone, *t, window, g.fps);
            }
            NX_PRINT << "Clef Operator: " << m_cameraName << ": window " << clock(window.startUs) << "-"
                << clock(window.endUs) << " " << g.priority << " " << fpsText(g.fps) << " fps " << toString(g.motion)
                << ": model " << fixed(result["model_seconds"].number_value()) << " s, queue "
                << fixed(result["queue_seconds"].number_value()) << " s";
        };
    std::vector<std::thread> senders;
    for (auto& [key, g]: groups)
        senders.emplace_back([&submit, &g]() { submit(g); });
    for (auto& sender: senders)
        sender.join();
}

void DeviceAgent::emitEvent(const QuestionConfig& q, const std::string& priority, const Zone* zone,
    const Transition& t, const Window& window, double fps)
{
    auto event = makePtr<EventMetadata>();
    event->setTypeId(eventTypeFor(priority));
    event->setCaption((zone ? zone->name + ": " : std::string()) + q.text);
    event->setDescription(std::string(t.open ? "Yes" : "Ended") + " · " + fixed(t.probability) + " · window "
        + clock(window.startUs) + "–" + clock(window.endUs) + " · " + fpsText(fps) + " fps · motion "
        + toString(window.motion.rank(m_engine->systemConfig()->activeMotionShare)));
    event->setIsActive(t.open);
    event->setConfidence((float) t.probability);
    event->setKey(m_cameraId + "/" + q.id);
    // Stamp the event with when it happened, not when Clef answered (up to a window later), so bookmarks and the
    // timeline land on the moment itself: the open at the frame with the most motion where the question looks,
    // the close at the end of the last window that still said yes.
    int64_t eventUs = m_lastFrameUs.load();
    if (t.open)
    {
        // Give the event a thumbnail: an object track whose best shot is the frame with the most motion where the
        // question looks, boxed around that motion. Clef has no boxes, so the motion grid supplies them.
        const Uuid trackId = UuidHelper::randomUuid();
        const BestShot shot = bestShot(window, zone ? &zone->cells : nullptr, zone ? &zone->box : nullptr);
        eventUs = shot.timestampUs;
        const Rect box((float) shot.x, (float) shot.y, (float) shot.w, (float) shot.h);

        auto object = makePtr<ObjectMetadata>();
        object->setTypeId(kAnswerObjectType);
        object->setTrackId(trackId);
        object->setBoundingBox(box);
        object->setConfidence((float) t.probability);
        object->addAttribute(makePtr<Attribute>("Question", q.text));
        object->addAttribute(makePtr<Attribute>("Answer", "Yes"));
        object->addAttribute(makePtr<Attribute>("Probability", fixed(t.probability)));
        object->addAttribute(makePtr<Attribute>("Zone", zone ? zone->name : std::string("Whole frame")));
        object->addAttribute(makePtr<Attribute>("Priority", priority));
        auto objectPacket = makePtr<ObjectMetadataPacket>();
        objectPacket->setTimestampUs(shot.timestampUs);
        objectPacket->setDurationUs(0);
        objectPacket->addItem(object);
        pushMetadataPacket(objectPacket);
        pushMetadataPacket(makePtr<ObjectTrackBestShotPacket>(trackId, shot.timestampUs, box));

        event->setTrackId(trackId);
        m_openEvents[q.id] = {eventTypeFor(priority), (zone ? zone->name + ": " : std::string()) + q.text,
            m_cameraId + "/" + q.id, trackId};
        NX_PRINT << "Clef Operator: " << m_cameraName << ": best shot " << clock(shot.timestampUs) << " box "
            << fixed(shot.x) << "," << fixed(shot.y) << " " << fixed(shot.w) << "x" << fixed(shot.h)
            << (shot.fromMotion ? " (motion)" : " (no motion: zone or frame)");
    }
    else
    {
        if (const auto open = m_openEvents.find(q.id); open != m_openEvents.end())
            event->setTrackId(open->second.trackId);
        m_openEvents.erase(q.id);
        eventUs = std::min(t.timeUs, eventUs);
    }

    auto packet = makePtr<EventMetadataPacket>();
    packet->setTimestampUs(eventUs);
    packet->setDurationUs(0);
    packet->addItem(event);
    pushMetadataPacket(packet);
    NX_PRINT << "Clef Operator: " << m_cameraName << ": EVENT " << (t.open ? "open " : "close ") << q.id << " \""
        << q.text << "\" p=" << fixed(t.probability);
}

void DeviceAgent::emitClosed(const OpenEvent& open, const std::string& reason)
{
    auto event = makePtr<EventMetadata>();
    event->setTypeId(open.typeId);
    event->setCaption(open.caption);
    event->setDescription("Ended · " + reason);
    event->setIsActive(false);
    event->setKey(open.key);
    event->setTrackId(open.trackId);
    auto packet = makePtr<EventMetadataPacket>();
    packet->setTimestampUs(m_lastFrameUs.load());
    packet->setDurationUs(0);
    packet->addItem(event);
    pushMetadataPacket(packet);
    NX_PRINT << "Clef Operator: " << m_cameraName << ": EVENT close \"" << open.caption << "\" (" << reason << ")";
}

void DeviceAgent::emitHealth(const std::string& caption, const std::string& description)
{
    const int64_t now = m_lastFrameUs.load();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (now - m_lastHealthUs < 300 * kSecondUs)
            return; //< At most one health event per camera every 5 minutes.
        m_lastHealthUs = now;
    }
    auto event = makePtr<EventMetadata>();
    event->setTypeId("nx.clef_operator.health");
    event->setCaption(caption);
    event->setDescription(description);
    event->setIsActive(true);
    auto packet = makePtr<EventMetadataPacket>();
    packet->setTimestampUs(now);
    packet->setDurationUs(0);
    packet->addItem(event);
    pushMetadataPacket(packet);
}

void DeviceAgent::post(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(m_taskMutex);
        if (m_tasks.size() > 20)
            m_tasks.pop_front(); //< Gateway far behind: drop the oldest window rather than grow without bound.
        m_tasks.push_back(std::move(task));
    }
    m_taskWake.notify_one();
}

void DeviceAgent::runTasks()
{
    for (;;)
    {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(m_taskMutex);
            m_taskWake.wait(lock, [this]() { return m_stopping || !m_tasks.empty(); });
            if (m_stopping)
                return;
            task = std::move(m_tasks.front());
            m_tasks.pop_front();
        }
        task();
    }
}

} // namespace clef_operator
