#include "ReIdConfigLoader.h"
#include <fstream>

namespace reid {

bool loadReIdConfig(const char* configPath, ReIdManagerConfig* config) {
    if (!configPath || !config) return false;

    std::ifstream f(configPath);
    if (!f) return false;

    nlohmann::json j;
    try {
        f >> j;
    } catch (...) {
        return false;
    }

    if (j.contains("enginePathDla0") && j["enginePathDla0"].is_string()) {
        config->enginePathDla0 = j["enginePathDla0"].get<std::string>();
    }
    if (j.contains("enginePathDla1") && j["enginePathDla1"].is_string()) {
        config->enginePathDla1 = j["enginePathDla1"].get<std::string>();
    }
    if (j.contains("engineHasPreprocess") && j["engineHasPreprocess"].is_boolean()) {
        config->conditionerConfig.engineHasPreprocess = j["engineHasPreprocess"].get<bool>();
    }
    return true;
}

}  // namespace reid
