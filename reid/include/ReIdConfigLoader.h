#ifndef REID_CONFIG_LOADER_H
#define REID_CONFIG_LOADER_H

#include <nlohmann/json.hpp>

#include "ReIdManager.h"

namespace reid {

bool loadReIdConfig(const char* configPath, ReIdManagerConfig* config);

}  // namespace reid

#endif  // REID_CONFIG_LOADER_H
