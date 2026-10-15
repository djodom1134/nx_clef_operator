#pragma once

#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <nx/sdk/analytics/helpers/engine.h>

#include "config.h"

namespace clef_operator {

/** Clef's choices for one question text (gateway /v1/classify). */
struct Classification
{
    double fps = 2;
    std::string priority = "normal";
    bool needsMotion = false;
    std::string summary;
};

/** A question with every "auto" field resolved. */
struct ResolvedQuestion
{
    QuestionConfig config;
    double fps = 2;
    std::string priority = "normal";
    bool needsMotion = false;
    bool classified = false; //< false while Clef's choice is still pending (defaults are used)
};

class Engine: public nx::sdk::analytics::Engine
{
public:
    Engine();
    virtual ~Engine() override;

    std::shared_ptr<const SystemConfig> systemConfig() const;

    /** Resolve "auto" fields from the classification cache; queue a classification if missing. */
    ResolvedQuestion resolve(const QuestionConfig& question, const std::string& cameraName,
        const std::string& cameraDescription);

    /** Clef's choice for display, or empty while pending. */
    std::string classificationSummary(const std::string& questionText) const;

protected:
    virtual std::string manifestString() const override;
    virtual nx::sdk::Result<const nx::sdk::ISettingsResponse*> settingsReceived() override;
    virtual void doObtainDeviceAgent(
        nx::sdk::Result<nx::sdk::analytics::IDeviceAgent*>* outResult,
        const nx::sdk::IDeviceInfo* deviceInfo) override;

private:
    void classifyLoop();

    mutable std::mutex m_mutex;
    std::shared_ptr<const SystemConfig> m_system = std::make_shared<SystemConfig>();
    std::map<std::string, Classification> m_classifications;
    std::set<std::string> m_pendingTexts;

    struct ClassifyRequest
    {
        std::string text;
        std::string camera;
        std::string description;
    };
    std::deque<ClassifyRequest> m_classifyQueue;
    std::condition_variable m_classifyWake;
    bool m_stopping = false;
    std::thread m_classifyThread;
};

} // namespace clef_operator
