#include "engine.h"

#include <cstdlib>

#include <nx/kit/debug.h>
#include <nx/kit/json.h>
#include <nx/sdk/helpers/settings_response.h>

#include "device_agent.h"
#include "http_client.h"

namespace clef_operator {

using namespace nx::sdk;
using namespace nx::sdk::analytics;
using nx::kit::Json;

const std::vector<std::pair<std::string, std::string>> kEventTypes = {
    {"nx.clef_operator.critical", "Clef: critical"},
    {"nx.clef_operator.high", "Clef: high"},
    {"nx.clef_operator.normal", "Clef: normal"},
    {"nx.clef_operator.low", "Clef: low / maintenance"},
};
const std::string kNuisanceEventType = "nx.clef_operator.nuisance";
const std::string kAnswerObjectType = "nx.clef_operator.answer";
const std::string kAnswersObjectType = "nx.clef_operator.answers"; //< the answers shown on the video
const std::string kHealthEventType = "nx.clef_operator.health";

Engine::Engine():
    nx::sdk::analytics::Engine(/*enableOutput*/ false),
    m_classifyThread([this]() { classifyLoop(); })
{
}

Engine::~Engine()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
    }
    m_classifyWake.notify_all();
    if (m_classifyThread.joinable())
        m_classifyThread.join();
}

std::string Engine::manifestString() const
{
    std::string error;
    Json::array eventTypes;
    for (const auto& [id, name]: kEventTypes)
        eventTypes.push_back(Json::object{{"id", id}, {"name", name}, {"flags", "stateDependent"}});
    eventTypes.push_back(Json::object{{"id", kNuisanceEventType}, {"name", "Clef: nuisance found"}});
    eventTypes.push_back(Json::object{{"id", kHealthEventType}, {"name", "Clef: system health"}});
    return Json(Json::object{
        {"capabilities", ""},
        {"streamTypeFilter", "compressedVideo|motion"},
        {"preferredStream", "secondary"},
        {"typeLibrary", Json::object{{"eventTypes", eventTypes}, {"objectTypes", Json::array{
            Json::object{{"id", kAnswerObjectType}, {"name", "Clef answer"}, {"attributes", Json::array{
                Json::object{{"type", "String"}, {"name", "Question"}},
                Json::object{{"type", "String"}, {"name", "Answer"}},
                Json::object{{"type", "Number"}, {"name", "Probability"}},
                Json::object{{"type", "String"}, {"name", "Zone"}},
                Json::object{{"type", "String"}, {"name", "Priority"}},
            }}},
            // Telemetry, not something to search for: shown live and in playback, kept out of Object Search.
            Json::object{{"id", kAnswersObjectType}, {"name", "Clef answers"}, {"flags", "nonIndexable"}},
        }}}},
        {"deviceAgentSettingsModel", Json::parse(deviceAgentSettingsModel(), error)},
    }).dump();
}

Result<const ISettingsResponse*> Engine::settingsReceived()
{
    auto config = std::make_shared<SystemConfig>(parseSystemConfig(currentSettings()));
    NX_PRINT << "Clef Operator: gateway " << config->gatewayUrl << ", " << config->questions.size()
        << " system questions, window " << config->windowSeconds << " s, overlap " << config->overlapSeconds << " s";
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_system = config;
    }
    for (const auto& q: config->questions)
        resolve(q, "", "");

    auto response = makePtr<SettingsResponse>();
    for (const auto& q: config->questions)
    {
        const std::string prefix = "sq" + q.id.substr(1);
        const std::string summary = classificationSummary(q.text);
        response->setValue(prefix + ".clef", summary.empty() ? "pending: reopen to see Clef's choice" : summary);
    }
    return response.releasePtr();
}

void Engine::doObtainDeviceAgent(Result<IDeviceAgent*>* outResult, const IDeviceInfo* deviceInfo)
{
    *outResult = new DeviceAgent(this, deviceInfo);
}

std::shared_ptr<const SystemConfig> Engine::systemConfig() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_system;
}

ResolvedQuestion Engine::resolve(
    const QuestionConfig& q, const std::string& cameraName, const std::string& cameraDescription)
{
    ResolvedQuestion r;
    r.config = q;
    const bool needsClef = q.fps == "auto" || q.priority == "auto" || q.needsMotion == "auto";
    Classification c;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_classifications.find(q.text);
        if (it != m_classifications.end())
        {
            c = it->second;
            r.classified = true;
        }
        else if (needsClef && !m_pendingTexts.count(q.text))
        {
            m_pendingTexts.insert(q.text);
            m_classifyQueue.push_back({q.text, cameraName, cameraDescription});
            m_classifyWake.notify_one();
        }
    }
    const double maxFps = systemConfig()->maxFps;
    r.fps = std::min(maxFps, q.fps == "auto" ? c.fps : std::atof(q.fps.c_str()));
    r.priority = q.priority == "auto" ? c.priority : q.priority;
    r.needsMotion = q.needsMotion == "auto" ? c.needsMotion : q.needsMotion == "yes";
    if (!needsClef)
        r.classified = true;
    return r;
}

std::string Engine::classificationSummary(const std::string& questionText) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_classifications.find(questionText);
    return it == m_classifications.end() ? "" : it->second.summary;
}

void Engine::classifyLoop()
{
    for (;;)
    {
        ClassifyRequest request;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_classifyWake.wait(lock, [this]() { return m_stopping || !m_classifyQueue.empty(); });
            if (m_stopping)
                return;
            request = m_classifyQueue.front();
            m_classifyQueue.pop_front();
        }
        const auto system = systemConfig();
        const std::string body = Json(Json::object{
            {"question", request.text}, {"camera", request.camera}, {"description", request.description},
            {"interests", system->siteInterests},
        }).dump();
        const auto response = httpPostJson(system->gatewayUrl + "/v1/classify", body, system->gatewayToken, 120);
        std::string error;
        const Json json = Json::parse(response.body, error);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pendingTexts.erase(request.text);
        if (response.status != 200 || !error.empty())
        {
            NX_PRINT << "Clef Operator: classify failed for \"" << request.text << "\": "
                << (response.error.empty() ? response.body.substr(0, 200) : response.error);
            continue;
        }
        Classification c;
        c.fps = json["fps"].number_value();
        c.priority = json["priority"].string_value();
        c.needsMotion = json["needs_motion"].bool_value();
        c.summary = json["summary"].string_value();
        m_classifications[request.text] = c;
        NX_PRINT << "Clef Operator: \"" << request.text << "\" -> " << c.summary;
    }
}

} // namespace clef_operator
