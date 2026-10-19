#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

#include <nx/kit/json.h>

namespace clef_operator {

using nx::kit::Json;

namespace {

Json textField(const std::string& name, const std::string& caption, const std::string& def = "",
    const std::string& description = "")
{
    Json::object o{{"type", "TextField"}, {"name", name}, {"caption", caption}, {"defaultValue", def}};
    if (!description.empty())
        o["description"] = description;
    return o;
}

Json comboBox(const std::string& name, const std::string& caption, const std::string& def,
    const std::vector<std::pair<std::string, std::string>>& items)
{
    Json::array range;
    Json::object captions;
    for (const auto& [value, label]: items)
    {
        range.push_back(value);
        captions[value] = label;
    }
    return Json::object{{"type", "ComboBox"}, {"name", name}, {"caption", caption}, {"defaultValue", def},
        {"range", range}, {"itemCaptions", captions}};
}

Json spinBox(const std::string& name, const std::string& caption, int def, int min, int max)
{
    return Json::object{{"type", "SpinBox"}, {"name", name}, {"caption", caption}, {"defaultValue", def},
        {"minValue", min}, {"maxValue", max}};
}

Json doubleSpinBox(const std::string& name, const std::string& caption, double def, double min, double max)
{
    return Json::object{{"type", "DoubleSpinBox"}, {"name", name}, {"caption", caption}, {"defaultValue", def},
        {"minValue", min}, {"maxValue", max}, {"decimals", 2}, {"step", 0.05}};
}

Json checkBox(const std::string& name, const std::string& caption, bool def)
{
    return Json::object{{"type", "CheckBox"}, {"name", name}, {"caption", caption}, {"defaultValue", def}};
}

Json group(const std::string& caption, const Json::array& items)
{
    return Json::object{{"type", "GroupBox"}, {"caption", caption}, {"items", items}};
}

/** Fields of one question; ``p`` is the name prefix with the Repeater's '#', e.g. "sq#". */
Json::array questionItems(const std::string& p, bool withZone, const std::vector<std::string>& zoneNames)
{
    Json::array items{
        checkBox(p + ".enabled", "Enabled", true),
        textField(p + ".text", "Question", "", "Plain language, e.g. \"Is anyone fighting?\""),
        comboBox(p + ".kind", "Answer type", "noul",
            {{"noul", "Yes / no"}, {"choice", "Choice"}, {"score", "Scale (ordered)"}}),
        textField(p + ".options", "Options", "", "Choice or scale only; comma-separated, scale from low to high"),
        textField(p + ".alertOn", "Alert on", "",
            "Choice: options that open an event. Scale: the lowest level that opens one"),
        comboBox(p + ".fps", "Frame rate", "auto",
            {{"auto", "Auto (Clef decides)"}, {"1", "1 fps"}, {"2", "2 fps"}, {"4", "4 fps"}, {"6", "6 fps"},
                {"8", "8 fps"}, {"10", "10 fps"}}),
        comboBox(p + ".priority", "Priority", "auto",
            {{"auto", "Auto (Clef decides)"}, {"critical", "Critical"}, {"high", "High"}, {"normal", "Normal"},
                {"low", "Low"}, {"maintenance", "Maintenance"}}),
        comboBox(p + ".needsMotion", "Needs motion", "auto",
            {{"auto", "Auto (Clef decides)"}, {"yes", "Yes: skip windows without motion"},
                {"no", "No: always ask"}}),
        doubleSpinBox(p + ".onThreshold", "Open event at", 0.8, 0.05, 1.0),
        doubleSpinBox(p + ".offThreshold", "Close event below", 0.5, 0.0, 1.0),
    };
    if (withZone)
    {
        std::vector<std::pair<std::string, std::string>> zones{{"0", "Whole frame"}};
        for (int i = 1; i <= kMaxZones; ++i)
        {
            const std::string name = i <= (int) zoneNames.size() && !zoneNames[i - 1].empty()
                ? zoneNames[i - 1] : "Zone " + std::to_string(i);
            zones.push_back({std::to_string(i), name});
        }
        items.push_back(comboBox(p + ".zone", "Zone", "0", zones));
    }
    // Filled by the plugin after Clef classifies the question; read-only.
    items.push_back(Json::object{{"type", "TextField"}, {"name", p + ".clef"}, {"caption", "Clef's choice"},
        {"defaultValue", ""}, {"enabled", false}});
    return items;
}

Json repeater(int count, const std::string& caption, const std::string& filledItem, const Json::array& items)
{
    return Json::object{{"type", "Repeater"}, {"count", count}, {"startIndex", 1},
        {"template", Json::object{{"type", "GroupBox"}, {"caption", caption},
            {"filledCheckItems", Json::array{filledItem}}, {"items", items}}}};
}

std::string get(const Settings& s, const std::string& key, const std::string& def = "")
{
    const auto it = s.find(key);
    return it == s.end() ? def : it->second;
}

double getDouble(const Settings& s, const std::string& key, double def)
{
    const std::string v = get(s, key);
    if (v.empty())
        return def;
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    return end == v.c_str() ? def : d;
}

bool getBool(const Settings& s, const std::string& key, bool def)
{
    const std::string v = get(s, key);
    return v.empty() ? def : v == "true";
}

std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
        return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

QuestionConfig parseQuestion(const Settings& s, const std::string& prefix, const std::string& id,
    double defaultOn, double defaultOff)
{
    QuestionConfig q;
    q.id = id;
    q.text = trim(get(s, prefix + ".text"));
    q.kind = get(s, prefix + ".kind", "noul");
    q.options = splitList(get(s, prefix + ".options"), ',');
    q.alertOn = splitList(get(s, prefix + ".alertOn"), ',');
    q.fps = get(s, prefix + ".fps", "auto");
    q.priority = get(s, prefix + ".priority", "auto");
    q.needsMotion = get(s, prefix + ".needsMotion", "auto");
    q.zone = std::atoi(get(s, prefix + ".zone", "0").c_str());
    q.onThreshold = getDouble(s, prefix + ".onThreshold", defaultOn);
    q.offThreshold = getDouble(s, prefix + ".offThreshold", defaultOff);
    return q;
}

} // namespace

std::vector<std::string> splitList(const std::string& text, char separator)
{
    std::vector<std::string> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, separator))
    {
        item = trim(item);
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

std::string engineSettingsModel()
{
    const Json model = Json::object{{"type", "Settings"}, {"items", Json::array{
        group("Inference", {
            textField("gatewayUrl", "Gateway URL", "http://192.168.87.134:8200"),
            Json::object{{"type", "PasswordField"}, {"name", "gatewayToken"}, {"caption", "Access token"},
                {"defaultValue", ""}},
        }),
        group("Site", {
            textField("siteInterests", "What matters here", "",
                "e.g. people or vehicles on the property after 22:00; anyone near the back door"),
            textField("nuisanceSources", "Known nuisance sources", "",
                "Separate with ';', e.g. flag on the driveway pole; trees along the back fence"),
        }),
        group("Windows", {
            spinBox("windowSeconds", "Window length (s)", 30, 10, 60),
            spinBox("overlapSeconds", "Overlap (s)", 10, 0, 25),
            checkBox("onsetWindow", "Early window on motion onset", true),
        }),
        group("Motion", {
            spinBox("activeMotionPercent", "Active motion (% of frames)", 5, 1, 100),
            spinBox("quietSeconds", "Quiet period before onset (s)", 60, 5, 600),
        }),
        group("Events and frame rate", {
            doubleSpinBox("onThreshold", "Default: open event at", 0.8, 0.05, 1.0),
            doubleSpinBox("offThreshold", "Default: close event below", 0.5, 0.0, 1.0),
            spinBox("closeAfter", "Windows below before an event closes", 1, 1, 10),
            comboBox("maxFps", "Maximum fps for any question", "10",
                {{"1", "1"}, {"2", "2"}, {"4", "4"}, {"6", "6"}, {"8", "8"}, {"10", "10"}}),
        }),
    }}, {"sections", Json::array{
        Json::object{{"type", "Section"}, {"name", "systemQuestions"}, {"caption", "System questions (all cameras)"},
            {"items", Json::array{
                repeater(kMaxSystemQuestions, "System question #", "sq#.text", questionItems("sq#", false, {})),
            }}},
    }}};
    return model.dump();
}

std::string deviceAgentSettingsModel(const std::vector<std::string>& zoneNames)
{
    const Json model = Json::object{{"type", "Settings"}, {"items", Json::array{
        group("Camera", {
            textField("cameraDescription", "About this camera", "",
                "e.g. School hallway; no one should be here after 22:00"),
            textField("extraNuisance", "Nuisance sources here", "", "Separate with ';'"),
            textField("systemQuestionsOff", "System questions off here", "", "Numbers, e.g. 1, 3"),
        }),
    }}, {"sections", Json::array{
        Json::object{{"type", "Section"}, {"name", "zones"}, {"caption", "Zones"}, {"items", Json::array{
            repeater(kMaxZones, "Zone #", "zone#", {
                Json::object{{"type", "PolygonFigure"}, {"name", "zone#"}, {"caption", "Shape (label = zone name)"},
                    {"minPoints", 3}, {"maxPoints", 12}},
                textField("zone#.description", "Description", ""),
            }),
        }}},
        Json::object{{"type", "Section"}, {"name", "ptzHunter"}, {"caption", "PTZ hunter"}, {"items", Json::array{
            Json::object{{"type", "Banner"}, {"icon", "info"}, {"text",
                "PTZ cameras only. Describe in plain words what to look for: the hunter keeps a map of what every "
                "direction looks like (an embedding of each look) and moves the camera slowly toward the directions "
                "that look most like it, away from what it should stay away from. Clef confirms on every look. "
                "Search the map from the Clef dashboard."}},
            checkBox("hunter.enabled", "Enable PTZ hunter", false),
            textField("hunter.targets", "Look for", kDefaultHunterTargets,
                "What to find, in plain words, separated with ';' (e.g. a person; a delivery van)"),
            textField("hunter.avoid", "Stay away from", "",
                "Things not worth watching, in plain words, separated with ';' (e.g. trees close to the camera; the "
                "neighbour's yard). Walls, roofs, the sky and leaves filling the view are always avoided"),
            textField("hunter.cues", "Look closer at", kDefaultHunterCues,
                "Places where what you look for tends to be, in plain words, separated with ';'. The hunter favours "
                "directions that look like one and zooms in on them"),
            textField("hunter.magnets", "Magnet views", "",
                "Preset numbers (e.g. 1, 2) or pan,tilt in degrees separated with ';'. Each preset is visited once "
                "to learn where it points. The camera's resting view is always a magnet"),
            doubleSpinBox("hunter.threshold", "Found at", 0.7, 0.3, 1.0),
            doubleSpinBox("hunter.maxSpeed", "Maximum motor speed", 0.1, 0.02, 1.0),
            spinBox("hunter.restSeconds", "Rest between moves (s)", 20, 0, 600),
            spinBox("hunter.dwellSeconds", "Dwell when found (s)", 60, 10, 1800),
            spinBox("hunter.widePercent", "Time zoomed all the way out (%)", 75, 0, 100),
            spinBox("hunter.focusSeconds", "Focus time after zooming (s)", 4, 0, 30),
            spinBox("hunter.handBackSeconds", "Take back after an operator or auto-tracking (s)", 10, 5, 3600),
            doubleSpinBox("hunter.panMin", "Pan from (degrees)", 0.0, -1.0, 360.0),
            doubleSpinBox("hunter.panMax", "Pan to (degrees)", 360.0, -1.0, 360.0),
            Json::object{{"type", "TextField"}, {"name", "hunter.status"}, {"caption", "Status"}, {"defaultValue", ""},
                {"enabled", false}},
        }}},
        Json::object{{"type", "Section"}, {"name", "cameraQuestions"}, {"caption", "Camera questions"},
            {"items", Json::array{
                repeater(kMaxCameraQuestions, "Question #", "q#.text", questionItems("q#", true, zoneNames)),
            }}},
    }}};
    return model.dump();
}

bool parsePolygon(const std::string& value, Zone* zone)
{
    if (value.empty())
        return false;
    std::string error;
    const Json json = Json::parse(value, error);
    if (!error.empty() || !json.is_object())
        return false;
    zone->name = json["label"].string_value();
    zone->points.clear();
    for (const auto& point: json["figure"]["points"].array_items())
    {
        const auto& xy = point.array_items();
        if (xy.size() == 2)
            zone->points.push_back({xy[0].number_value(), xy[1].number_value()});
    }
    if (zone->points.size() < 3)
        return false;

    double x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const auto& [x, y]: zone->points)
    {
        x0 = std::min(x0, x); y0 = std::min(y0, y);
        x1 = std::max(x1, x); y1 = std::max(y1, y);
    }
    zone->box = {x0, y0, x1 - x0, y1 - y0};

    const auto inside =
        [&](double px, double py)
        {
            bool in = false;
            const auto& p = zone->points;
            for (size_t i = 0, j = p.size() - 1; i < p.size(); j = i++)
            {
                if ((p[i].second > py) != (p[j].second > py)
                    && px < (p[j].first - p[i].first) * (py - p[i].second) / (p[j].second - p[i].second) + p[i].first)
                {
                    in = !in;
                }
            }
            return in;
        };
    for (int y = 0; y < kGridH; ++y)
    {
        for (int x = 0; x < kGridW; ++x)
            zone->cells[y * kGridW + x] = inside((x + 0.5) / kGridW, (y + 0.5) / kGridH);
    }
    return true;
}

SystemConfig parseSystemConfig(const Settings& s)
{
    SystemConfig c;
    c.gatewayUrl = trim(get(s, "gatewayUrl", c.gatewayUrl));
    while (!c.gatewayUrl.empty() && c.gatewayUrl.back() == '/')
        c.gatewayUrl.pop_back();
    c.gatewayToken = get(s, "gatewayToken");
    c.siteInterests = trim(get(s, "siteInterests"));
    c.nuisanceSources = splitList(get(s, "nuisanceSources"), ';');
    c.windowSeconds = (int) getDouble(s, "windowSeconds", 30);
    c.overlapSeconds = std::min((int) getDouble(s, "overlapSeconds", 10), c.windowSeconds - 5);
    c.onsetWindow = getBool(s, "onsetWindow", true);
    c.activeMotionShare = getDouble(s, "activeMotionPercent", 5) / 100.0;
    c.quietSeconds = (int) getDouble(s, "quietSeconds", 60);
    c.onThreshold = getDouble(s, "onThreshold", 0.8);
    c.offThreshold = getDouble(s, "offThreshold", 0.5);
    c.closeAfter = std::max(1, (int) getDouble(s, "closeAfter", 1));
    c.maxFps = getDouble(s, "maxFps", 10);
    for (int i = 1; i <= kMaxSystemQuestions; ++i)
    {
        const std::string prefix = "sq" + std::to_string(i);
        auto q = parseQuestion(s, prefix, "s" + std::to_string(i), c.onThreshold, c.offThreshold);
        q.zone = 0;
        if (!q.text.empty() && getBool(s, prefix + ".enabled", true))
            c.questions.push_back(q);
    }
    return c;
}

CameraConfig parseCameraConfig(const Settings& s, const SystemConfig& system)
{
    CameraConfig c;
    c.description = trim(get(s, "cameraDescription"));
    c.extraNuisance = splitList(get(s, "extraNuisance"), ';');
    for (const auto& n: splitList(get(s, "systemQuestionsOff"), ','))
        c.systemQuestionsOff.push_back(std::atoi(n.c_str()));
    for (int i = 1; i <= kMaxZones; ++i)
    {
        Zone zone;
        zone.index = i;
        if (parsePolygon(get(s, "zone" + std::to_string(i)), &zone))
        {
            zone.description = trim(get(s, "zone" + std::to_string(i) + ".description"));
            if (zone.name.empty())
                zone.name = "Zone " + std::to_string(i);
            c.zones.push_back(zone);
        }
    }
    c.hunter.enabled = getBool(s, "hunter.enabled", false);
    c.hunter.targets = splitList(get(s, "hunter.targets", kDefaultHunterTargets), ';');
    c.hunter.avoid = splitList(get(s, "hunter.avoid"), ';');
    c.hunter.cues = splitList(get(s, "hunter.cues", kDefaultHunterCues), ';');
    // Magnets: preset numbers ("1, 2", "preset 3") or "pan,tilt[,fov]" in degrees. A field of only whole numbers
    // is a preset list: operators type the presets they use.
    const std::string magnetText = get(s, "hunter.magnets");
    const auto isInteger =
        [](const std::string& t) { return !t.empty() && t.find_first_not_of("0123456789") == std::string::npos; };
    bool onlyIntegers = true;
    for (const auto& entry: splitList(magnetText, ';'))
    {
        for (const auto& token: splitList(entry, ','))
            onlyIntegers = onlyIntegers && isInteger(token);
    }
    for (const auto& entry: splitList(magnetText, ';'))
    {
        const auto parts = splitList(entry, ',');
        if (onlyIntegers)
        {
            c.hunter.presets.insert(c.hunter.presets.end(), parts.begin(), parts.end());
        }
        else if (entry.rfind("preset", 0) == 0)
        {
            c.hunter.presets.push_back(splitList(entry.substr(6), ' ').empty() ? "" : splitList(entry.substr(6), ' ')[0]);
        }
        else if (parts.size() >= 2)
        {
            c.hunter.magnets.push_back({std::atof(parts[0].c_str()), std::atof(parts[1].c_str()),
                parts.size() > 2 ? std::atof(parts[2].c_str()) : 0.0});
        }
    }
    c.hunter.threshold = getDouble(s, "hunter.threshold", 0.7);
    c.hunter.maxSpeed = getDouble(s, "hunter.maxSpeed", 0.1);
    c.hunter.restSeconds = (int) getDouble(s, "hunter.restSeconds", 20);
    c.hunter.dwellSeconds = (int) getDouble(s, "hunter.dwellSeconds", 60);
    c.hunter.widePercent = (int) getDouble(s, "hunter.widePercent", 75);
    c.hunter.focusSeconds = (int) getDouble(s, "hunter.focusSeconds", 4);
    c.hunter.handBackSeconds = (int) getDouble(s, "hunter.handBackSeconds", 10);
    c.hunter.panMin = getDouble(s, "hunter.panMin", 0.0);
    c.hunter.panMax = getDouble(s, "hunter.panMax", 360.0);
    if (std::fabs(c.hunter.panMin) <= 1.0 && std::fabs(c.hunter.panMax) <= 1.0 && c.hunter.panMax > c.hunter.panMin)
    {
        // Saved before the range was in degrees: Nx normalized pan, -1..1 = 0..360 degrees.
        c.hunter.panMin = 180.0 * (c.hunter.panMin + 1.0);
        c.hunter.panMax = 180.0 * (c.hunter.panMax + 1.0);
    }

    for (int i = 1; i <= kMaxCameraQuestions; ++i)
    {
        const std::string prefix = "q" + std::to_string(i);
        auto q = parseQuestion(s, prefix, "c" + std::to_string(i), system.onThreshold, system.offThreshold);
        if (!q.text.empty() && getBool(s, prefix + ".enabled", true))
            c.questions.push_back(q);
    }
    return c;
}

} // namespace clef_operator
