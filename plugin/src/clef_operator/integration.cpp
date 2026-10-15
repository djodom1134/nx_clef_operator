#include "integration.h"

#include <nx/kit/json.h>

#include "config.h"
#include "engine.h"

namespace clef_operator {

using namespace nx::sdk;
using namespace nx::sdk::analytics;

Result<IEngine*> Integration::doObtainEngine()
{
    return new Engine();
}

std::string Integration::manifestString() const
{
    std::string error;
    const nx::kit::Json settingsModel = nx::kit::Json::parse(engineSettingsModel(), error);
    return nx::kit::Json(nx::kit::Json::object{
        {"id", "nx.clef_operator"},
        {"name", "Clef Operator"},
        {"description", "Ask plain-language questions about camera video. Clef answers them continuously "
            "on a local GPU and each answer arrives as an analytics event."},
        {"version", "0.1.0"},
        {"vendor", "Network Optix"},
        {"engineSettingsModel", settingsModel},
    }).dump();
}

} // namespace clef_operator
