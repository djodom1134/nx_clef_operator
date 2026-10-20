// Tests for the plugin parts that do not need the Server. Run: ./clef_operator_tests
#include <cassert>
#include <cmath>
#include <cstdio>

#include <nx/kit/json.h>

#include "clef_operator/clip_buffer.h"
#include "clef_operator/config.h"
#include "clef_operator/event_state.h"
#include "clef_operator/http_client.h"

using namespace clef_operator;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static void testBase64()
{
    CHECK(base64Encode("") == "");
    CHECK(base64Encode("f") == "Zg==");
    CHECK(base64Encode("fo") == "Zm8=");
    CHECK(base64Encode("foo") == "Zm9v");
    CHECK(base64Encode(std::string("\0\1\2", 3)) == "AAEC");
}

static void testModelsAreValidJson()
{
    std::string error;
    const auto engine = nx::kit::Json::parse(engineSettingsModel(), error);
    CHECK(error.empty() && engine["type"].string_value() == "Settings");
    const auto camera = nx::kit::Json::parse(deviceAgentSettingsModel({"Back door"}), error);
    CHECK(error.empty() && camera["sections"].array_items().size() == 3);
    CHECK(deviceAgentSettingsModel({"Back door"}).find("Back door") != std::string::npos);
}

static void testParseConfig()
{
    Settings s{
        {"gatewayUrl", "http://10.0.0.5:8200/"}, {"nuisanceSources", "flag on pole; trees ;"},
        {"sq1.text", "Is anyone fighting?"}, {"sq1.priority", "critical"}, {"sq1.fps", "8"},
        {"sq2.text", ""}, {"sq3.text", "Is the lens dirty?"}, {"sq3.enabled", "false"},
    };
    const auto system = parseSystemConfig(s);
    CHECK(system.gatewayUrl == "http://10.0.0.5:8200");
    CHECK(system.nuisanceSources.size() == 2 && system.nuisanceSources[1] == "trees");
    CHECK(system.questions.size() == 1 && system.questions[0].id == "s1" && system.questions[0].fps == "8");

    Settings c{
        {"zone2", R"({"figure":{"color":"#fff","points":[[0.5,0.5],[1.0,0.5],[1.0,1.0],[0.5,1.0]]},"label":"Back door"})"},
        {"q1.text", "Is someone at the door?"}, {"q1.zone", "2"}, {"systemQuestionsOff", "1, 3"},
        {"hunter.enabled", "true"}, {"hunter.targets", "Is there a person in view?; Is a car parked?"},
        {"hunter.magnets", "184,-12; 248,-41.5,20; bad"}, {"hunter.maxSpeed", "0.05"},
        {"hunter.panMin", "-0.6"}, {"hunter.panMax", "0.6"},
    };
    const auto camera = parseCameraConfig(c, system);
    CHECK(camera.zones.size() == 1 && camera.zones[0].name == "Back door" && camera.zones[0].index == 2);
    CHECK(std::fabs(camera.zones[0].box[0] - 0.5) < 1e-9 && std::fabs(camera.zones[0].box[2] - 0.5) < 1e-9);
    CHECK(camera.zones[0].cells[31 * kGridW + 43] && !camera.zones[0].cells[0]);
    CHECK(camera.questions.size() == 1 && camera.questions[0].zone == 2);
    CHECK(camera.systemQuestionsOff.size() == 2 && camera.systemQuestionsOff[1] == 3);
    CHECK(camera.hunter.enabled && camera.hunter.targets.size() == 2 && camera.hunter.maxSpeed == 0.05);
    CHECK(camera.hunter.magnets.size() == 2 && camera.hunter.magnets[1][2] == 20 && camera.hunter.presets.empty());
    CHECK(std::fabs(camera.hunter.panMin - 72.0) < 1e-9 && std::fabs(camera.hunter.panMax - 288.0) < 1e-9);
    Settings presets{{"hunter.magnets", "1,2"}};
    const auto p = parseCameraConfig(presets, system);
    CHECK(p.hunter.presets == std::vector<std::string>({"1", "2"}) && p.hunter.magnets.empty());
}

static void testClipBufferCut()
{
    ClipBuffer b;
    const std::string sps("\0\0\0\1\x67", 5);
    for (int i = 0; i < 70; ++i) //< 35 s at 2 fps, a keyframe every 4 s
    {
        const int64_t ts = i * 500'000LL;
        b.addVideo(ts, i % 8 == 0, "h264", std::string("\0\0\0\1\x65", 5), sps);
        CellMask cells{};
        cells[5] = i >= 60;
        b.addMotion(ts, cells, i >= 60);
    }
    Window w;
    CHECK(b.cut(34'000'000, 30'000'000, &w));
    CHECK(w.startUs == 4'000'000); //< keyframe at or before 4 s
    CHECK(w.frames == 61);
    CHECK(w.annexB.compare(0, 5, sps) == 0);
    CHECK(w.motion.framesWithMotion == 9 && w.motion.heat[5] == 9);
    CHECK(w.motion.rank(0.05) == MotionRank::active);
    CellMask elsewhere{};
    elsewhere[100] = true;
    CHECK(w.motion.rank(0.05, &elsewhere) == MotionRank::still);
}

static void testBestShot()
{
    Window w;
    w.frameTimestampsUs = {0, 500'000, 1'000'000};
    CellMask few{}, many{};
    few[5] = true;
    for (int x = 10; x < 14; ++x)
        many[2 * kGridW + x] = true;
    w.motionFrames = {{0, few}, {500'000, many}};
    BestShot s = bestShot(w, nullptr, nullptr);
    CHECK(s.fromMotion && s.timestampUs == 500'000);
    CHECK(std::fabs(s.x - 9.0 / kGridW) < 1e-9 && std::fabs(s.w - 6.0 / kGridW) < 1e-9);
    CHECK(std::fabs(s.y - 1.0 / kGridH) < 1e-9 && std::fabs(s.h - 3.0 / kGridH) < 1e-9);

    CellMask zone{};
    zone[5] = true; //< only the cell that moved in the first frame
    s = bestShot(w, &zone, nullptr);
    CHECK(s.fromMotion && s.timestampUs == 0);

    const std::array<double, 4> box{0.5, 0.5, 0.25, 0.25};
    Window still;
    still.frameTimestampsUs = {0, 500'000, 1'000'000};
    s = bestShot(still, &zone, &box);
    CHECK(!s.fromMotion && s.timestampUs == 500'000 && s.x == 0.5 && s.w == 0.25);
}

static void testZoneMotionCountsFastCrossings()
{
    // A car crossing a 10-cell road zone in 2 s at 7 fps: each cell moves in only 1-2 of 210 frames.
    Window w;
    w.startUs = 0;
    w.endUs = 30'000'000;
    w.fps = 7;
    w.motion.frames = 210;
    CellMask road{};
    for (int x = 0; x < 10; ++x)
        road[5 * kGridW + x] = true;
    for (int f = 0; f < 14; ++f)
    {
        CellMask cells{};
        cells[5 * kGridW + (f * 10 / 14)] = true;
        w.motionFrames.push_back({f * 142'857LL, cells});
        ++w.motion.framesWithMotion;
        ++w.motion.heat[5 * kGridW + (f * 10 / 14)];
    }
    CHECK(w.motion.rank(0.05, &road) == MotionRank::still); //< the old busiest-cell measure missed it
    CHECK(framesWithMotionIn(w, &road) == 14);
    CHECK(zoneMotionRank(w, &road, 0.05) == MotionRank::active);
    CellMask elsewhere{};
    elsewhere[0] = true;
    CHECK(zoneMotionRank(w, &elsewhere, 0.05) == MotionRank::still);
    CHECK(zoneMotionRank(w, nullptr, 0.08) == MotionRank::active); //< whole frame: 14 frames in a row = 2 s of car

    // The 06:01 car: 10 consecutive frames of 213 (under 5%) across the whole frame must count.
    Window car;
    car.startUs = 0; car.endUs = 30'000'000; car.fps = 7; car.motion.frames = 213;
    for (int f = 0; f < 10; ++f)
    {
        CellMask cells{};
        cells[3 * kGridW + 20 + f] = true;
        car.motionFrames.push_back({10'000'000 + f * 142'857LL, cells});
        ++car.motion.framesWithMotion;
    }
    CHECK(longestMotionRunUs(car, nullptr) >= 1'000'000);
    CHECK(zoneMotionRank(car, nullptr, 0.05) == MotionRank::active);

    // Noise: 10 scattered single frames, none next to another, must not wake the questions.
    Window noise;
    noise.startUs = 0; noise.endUs = 30'000'000; noise.fps = 7; noise.motion.frames = 213;
    for (int f = 0; f < 10; ++f)
    {
        CellMask cells{};
        cells[(f * 97) % (kGridW * kGridH)] = true;
        noise.motionFrames.push_back({f * 2'900'000LL, cells});
        ++noise.motion.framesWithMotion;
    }
    CHECK(zoneMotionRank(noise, nullptr, 0.05) == MotionRank::still);
}

static void testAnnexB()
{
    const std::string avcc("\0\0\0\2\x65\x01\0\0\0\1\x41", 11);
    CHECK(toAnnexB(avcc) == std::string("\0\0\0\1\x65\x01\0\0\0\1\x41", 11));
}

static void testEventState()
{
    QuestionConfig q;
    EventState e;
    CHECK(!e.update(0, 30, 0.1, 0.8, 0.5, 1));
    auto t = e.update(20, 50, 0.9, 0.8, 0.5, 1);
    CHECK(t && t->open && t->timeUs == 20);
    CHECK(!e.update(40, 70, 0.6, 0.8, 0.5, 1));
    CHECK(!e.update(30, 60, 0.0, 0.8, 0.5, 1)); //< out of order: ignored
    t = e.update(60, 90, std::nullopt, 0.8, 0.5, 1);
    CHECK(t && !t->open && t->timeUs == 70 && t->probability == 0.9);

    QuestionConfig crowd;
    crowd.kind = "score";
    crowd.options = {"empty", "light", "busy", "packed"};
    crowd.alertOn = {"busy"};
    CHECK(std::fabs(alertProbability(crowd, {{"empty", 0.1}, {"light", 0.2}, {"busy", 0.3}, {"packed", 0.4}}) - 0.7) < 1e-9);
    CHECK(std::fabs(alertProbability(crowd, {{"0", 0.1}, {"1", 0.2}, {"2", 0.3}, {"3", 0.4}}) - 0.7) < 1e-9);
}

static void testHunterCues()
{
    const auto defaults = parseCameraConfig(Settings{}, SystemConfig());
    CHECK(defaults.hunter.cues.size() == 4 && defaults.hunter.cues[1] == "a gate");
    CHECK(defaults.hunter.targets.size() == 1 && defaults.hunter.targets[0] == "a person");
    CHECK(defaults.hunter.cameraHeight == 3.5);
    CHECK(defaults.hunter.focusSeconds == 4 && defaults.hunter.handBackSeconds == 10);
    const auto custom = parseCameraConfig(Settings{{"hunter.cues", "Is there a bench?; Is there a door?"}}, SystemConfig());
    CHECK(custom.hunter.cues.size() == 2 && custom.hunter.cues[1] == "Is there a door?");
}

int main()
{
    testBase64();
    testModelsAreValidJson();
    testParseConfig();
    testClipBufferCut();
    testAnnexB();
    testBestShot();
    testZoneMotionCountsFastCrossings();
    testEventState();
    testHunterCues();
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
