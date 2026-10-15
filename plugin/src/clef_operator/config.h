#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

namespace clef_operator {

constexpr int kGridW = 44;
constexpr int kGridH = 32;
constexpr int kMaxSystemQuestions = 8;
constexpr int kMaxCameraQuestions = 10;
constexpr int kMaxZones = 8;

using Settings = std::map<std::string, std::string>;
using CellMask = std::array<bool, kGridW * kGridH>;

/** A question as configured on the system or camera page. "auto" fields are filled by the classifier. */
struct QuestionConfig
{
    std::string id; //< "s3" for system question 3, "c2" for camera question 2.
    std::string text;
    std::string kind = "noul"; //< noul | choice | score
    std::vector<std::string> options;
    std::vector<std::string> alertOn;
    std::string fps = "auto"; //< "auto" or a number
    std::string priority = "auto"; //< "auto" | critical | high | normal | low | maintenance
    std::string needsMotion = "auto"; //< "auto" | yes | no
    int zone = 0; //< 0 = whole frame, else 1..kMaxZones
    double onThreshold = 0.8;
    double offThreshold = 0.5;
};

struct Zone
{
    int index = 0;
    std::string name;
    std::string description;
    std::vector<std::pair<double, double>> points; //< Normalized polygon.
    CellMask cells{}; //< Motion grid cells whose centre is inside the polygon.
    std::array<double, 4> box{}; //< Normalized x, y, w, h of the polygon.
};

struct SystemConfig
{
    std::string gatewayUrl = "http://192.168.87.134:8200";
    std::string gatewayToken;
    std::string siteInterests;
    std::vector<std::string> nuisanceSources;
    int windowSeconds = 30;
    int overlapSeconds = 10;
    bool onsetWindow = true;
    double activeMotionShare = 0.05;
    int quietSeconds = 60;
    double onThreshold = 0.8;
    double offThreshold = 0.5;
    int closeAfter = 1;
    double maxFps = 10;
    bool showAnswers = true; //< the latest answer to each question, small in the lower right of the video
    std::vector<QuestionConfig> questions;
};

inline const char* const kDefaultHunterCues = "Is there a path or walkway?; Is there a gate?; "
    "Is there a play area or play equipment?; Is there a picnic table, patio or seating area?; Is there a yard?";

struct HunterSettings
{
    bool enabled = false;
    std::vector<std::string> targets;
    std::vector<std::string> avoid;
    std::vector<std::string> cues; //< helper questions: places where the hunt questions are more often yes
    std::vector<std::array<double, 3>> magnets; //< pan, tilt, field of view (degrees; fov 0 = widest)
    std::vector<std::string> presets; //< preset ids used as magnets
    double threshold = 0.7;
    double maxSpeed = 0.1;
    int restSeconds = 20;
    int dwellSeconds = 60;
    int widePercent = 75; //< share of the time spent zoomed all the way out; the rest, looking closer at spots
    int focusSeconds = 4; //< after a zoom change, no commands or snapshots while the camera refocuses
    double panMin = 0.0; //< degrees
    double panMax = 360.0;
};

struct CameraConfig
{
    std::string description;
    std::vector<std::string> extraNuisance;
    std::vector<int> systemQuestionsOff; //< 1-based system question numbers turned off here.
    std::vector<Zone> zones;
    std::vector<QuestionConfig> questions;
    HunterSettings hunter;
};

std::string engineSettingsModel();
std::string deviceAgentSettingsModel(const std::vector<std::string>& zoneNames = {});

SystemConfig parseSystemConfig(const Settings& settings);
CameraConfig parseCameraConfig(const Settings& settings, const SystemConfig& system);

/** Parse a PolygonFigure value; empty if the figure has fewer than 3 points. */
bool parsePolygon(const std::string& value, Zone* zone);

std::vector<std::string> splitList(const std::string& text, char separator);

} // namespace clef_operator
