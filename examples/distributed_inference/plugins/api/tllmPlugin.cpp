/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "api/tllmPlugin.h"

#include "common/stringUtils.h"
#include "common/tllmLogger.h"

#include "ncclPlugin/allgatherPlugin.h"
#include "ncclPlugin/allreducePlugin.h"
#include <array>
#include <cstdlib>

#include <NvInferRuntime.h>

namespace tc = tensorrt_llm::common;

namespace {

template <typename T> class PluginRegistrarTRTLLM {
public:
  PluginRegistrarTRTLLM() { getPluginRegistry()->registerCreator(instance, "tensorrt_llm"); }

private:
  //! Plugin instance.
  T instance{};
};

static PluginRegistrarTRTLLM<tensorrt_llm::plugins::AllgatherPluginCreator>
    allgatherPluginCreatorRegistrar{};
static PluginRegistrarTRTLLM<tensorrt_llm::plugins::AllreducePluginCreator>
    allreducePluginCreatorRegistrar{};

nvinfer1::IPluginCreator *creatorPtr(nvinfer1::IPluginCreator &creator) { return &creator; }

auto tllmLogger = tensorrt_llm::runtime::TllmLogger();

nvinfer1::ILogger *gLogger{&tllmLogger};

class GlobalLoggerFinder : public nvinfer1::ILoggerFinder {
public:
  nvinfer1::ILogger *findLogger() override { return gLogger; }
};

GlobalLoggerFinder gGlobalLoggerFinder{};

#if !defined(_MSC_VER)
[[maybe_unused]] __attribute__((constructor))
#endif
void initOnLoad()
{
  auto constexpr kLoadPlugins = "TRT_LLM_LOAD_PLUGINS";
  auto const loadPlugins = std::getenv(kLoadPlugins);
  if (loadPlugins && loadPlugins[0] == '1') {
    initTrtLlmPlugins(gLogger);
  }
}

bool pluginsInitialized = false;

} // namespace

namespace tensorrt_llm::plugins::api {

LoggerManager &tensorrt_llm::plugins::api::LoggerManager::getInstance() noexcept {
  static LoggerManager instance;
  return instance;
}

void LoggerManager::setLoggerFinder(nvinfer1::ILoggerFinder *finder) {
  std::lock_guard<std::mutex> lk(mMutex);
  if (mLoggerFinder == nullptr && finder != nullptr) {
    mLoggerFinder = finder;
  }
}

[[maybe_unused]] nvinfer1::ILogger *LoggerManager::logger() {
  std::lock_guard<std::mutex> lk(mMutex);
  if (mLoggerFinder != nullptr) {
    return mLoggerFinder->findLogger();
  }
  return nullptr;
}

nvinfer1::ILogger *LoggerManager::defaultLogger() noexcept { return gLogger; }
} // namespace tensorrt_llm::plugins::api

// New Plugin APIs

extern "C" {
bool initTrtLlmPlugins(void *logger, char const *libNamespace) {
  if (pluginsInitialized)
    return true;

  if (logger) {
    gLogger = static_cast<nvinfer1::ILogger *>(logger);
  }
  setLoggerFinder(&gGlobalLoggerFinder);

  auto registry = getPluginRegistry();
  std::int32_t nbCreators;
  auto creators = getPluginCreators(nbCreators);

  for (std::int32_t i = 0; i < nbCreators; ++i) {
    auto const creator = creators[i];
    creator->setPluginNamespace(libNamespace);
    registry->registerCreator(*creator, libNamespace);
    if (gLogger) {
      auto const msg =
          tc::fmtstr("Registered plugin creator %s version %s in namespace %s",
                     creator->getPluginName(), creator->getPluginVersion(), libNamespace);
      gLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, msg.c_str());
    }
  }

  pluginsInitialized = true;
  return true;
}

[[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder *finder) {
  tensorrt_llm::plugins::api::LoggerManager::getInstance().setLoggerFinder(finder);
}

[[maybe_unused]] nvinfer1::IPluginCreator *const *getPluginCreators(std::int32_t &nbCreators) {
  static tensorrt_llm::plugins::AllgatherPluginCreator allgatherPluginCreator;
  static std::array pluginCreators = {
      creatorPtr(allgatherPluginCreator),
  };
  nbCreators = pluginCreators.size();
  return pluginCreators.data();
}

} // extern "C"
