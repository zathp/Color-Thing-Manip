#pragma once
#include "GuiManager.hpp"

// --- GuiManager ---
GuiManager::GuiManager() {

}
GuiManager::~GuiManager() = default;

void GuiManager::draw() {
    ImGui::Begin("Control Panel");

    // Group Selection
    static int selectedIndex = 0;
    std::vector<std::string> names;
    for (const auto& manager : *agentManagerList) {
        names.push_back(manager.getName());
    }

    if (ImGui::BeginCombo("Agent Group", names[selectedIndex].c_str())) {
        for (int i = 0; i < names.size(); ++i) {
            if (ImGui::Selectable(names[i].c_str(), i == selectedIndex))
                selectedIndex = i;
        }
        ImGui::EndCombo();
    }

    auto settings = agentManagerList->at(selectedIndex).getSettings();

    // ---- Group Settings Section ----
    if (ImGui::CollapsingHeader("Group Settings", ImGuiTreeNodeFlags_DefaultOpen)) {

        if (ImGui::TreeNode("Agent Settings")) {
            settings->Agents.agentCount.draw();
            settings->Agents.speed.draw();
            settings->Agents.turnFactor.draw();
            settings->Agents.randFactor.draw();
            settings->Agents.biasFactor.draw();
            settings->Agents.repulsion.draw();
            settings->Agents.lockDir.draw();
            settings->Agents.lockDirCount.draw();
            settings->Agents.memoryFactor.draw();
            settings->Agents.interpolateColorDrop.draw();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Search Settings")) {
            settings->Agents.Search.searchSize.draw();
            settings->Agents.Search.searchAngle.draw();
            settings->Agents.Search.searchAngleOffset.draw();
            settings->Agents.Search.searchMethodIndex.draw();
            settings->Agents.Search.useCuda.draw();
            AgentManager::searchPreview.drawGuiImage();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Color Settings")) {
            settings->Agents.Color.filterR.draw();
            settings->Agents.Color.filterG.draw();
            settings->Agents.Color.filterB.draw();
            settings->Agents.Color.hueRotationEnabled.draw();
            if (settings->Agents.Color.hueRotationEnabled) {
                settings->Agents.Color.hueRotationAngle.draw();
                settings->Agents.Color.hueOscillation.draw();
                settings->Agents.Color.hueOscillationPeriod.draw();
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Oscillation Settings")) {
            settings->Oscillation.timeOscillationEnabled.draw();
            settings->Oscillation.globalPeriodR.draw();
            settings->Oscillation.globalPeriodG.draw();
            settings->Oscillation.globalPeriodB.draw();
            settings->Oscillation.distAlternate.draw();
            settings->Oscillation.distR.draw();
            settings->Oscillation.distG.draw();
            settings->Oscillation.distB.draw();
            settings->Oscillation.distPeriodR.draw();
            settings->Oscillation.distPeriodG.draw();
            settings->Oscillation.distPeriodB.draw();
            ImGui::TreePop();
        }
    }

    // ---- Global Settings Section ----
    if (ImGui::CollapsingHeader("Global Audio Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        Settings->Audio.deviceIndex.draw();
        Settings->Audio.audioAlternate.draw();
        Settings->Audio.autoGainEqualizer.draw();
        Settings->Audio.gainR.draw();
        Settings->Audio.gainG.draw();
        Settings->Audio.gainB.draw();
        Settings->Audio.peakDriftEnabled.draw();
        Settings->Audio.peakDriftStrength.draw();
        Settings->Audio.centroidDriftEnabled.draw();
        Settings->Audio.centroidDriftStrength.draw();
        Settings->Audio.suppressLowPeakDrift.draw();
        Settings->Audio.suppressPeakStrength.draw();
        Settings->Audio.suppressLowCentroidDrift.draw();
        Settings->Audio.suppressCentroidStrength.draw();
    }

    if (ImGui::CollapsingHeader("Global Shader Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        Settings->Shader.dimRate.draw();
        Settings->Shader.disperseFactor.draw();
        Settings->Shader.useColorMap.draw();
        Settings->Shader.colorMapIndex.draw();
    }

    if (ImGui::CollapsingHeader("Global Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
        Settings->targetFPS.draw();
        Settings->targetUPS.draw();
    }

    ImGui::Separator();
    ImGui::Text("Audio Visualizer");
    audioManager->drawVisualizerImage();

    ImGui::End();
}


void GuiManager::init() {
    Settings->Shader.dimRate.setInputFunc([](float f) { return f / 100.0f; });
    Settings->Shader.dimRate.setOnChange([this] {
        this->recursiveShader->setUniform("dimRate", Settings->Shader.dimRate);
        });
    Settings->Shader.dimRate.setScaleSettings(-5.0f, 5.0f);

    Settings->Shader.disperseFactor.setOnChange([this] {
        this->recursiveShader->setUniform("disperseFactor", Settings->Shader.disperseFactor);
        });
    Settings->Shader.disperseFactor.setScaleSettings(0.0f, 2.0f);

    Settings->Shader.colorMapIndex.updateChoiceOptions(AgentManager::colorMapNames);

    Settings->Shader.colorMapIndex.setOnChange([&] {
        worldShader->setUniform("colorMatrix", AgentManager::getColorMapGlsl());
    });

    Settings->Shader.useColorMap.setOnChange([&] {
        worldShader->setUniform("colorMapEnabled", Settings->Shader.useColorMap);
    });

	Settings->Audio.deviceIndex.updateChoiceOptions(audioManager->getDeviceNames());
    Settings->Audio.deviceIndex.setOnChange([this] {
        this->audioManager->setDeviceIndex(Settings->Audio.deviceIndex, true);
    });

    Settings->Audio.audioAlternate.setOnChange([this] {
        this->audioManager->enableAudio(Settings->Audio.audioAlternate);
    });

    Settings->Audio.suppressPeakStrength.setScaleSettings(-2.0f, 2.0f);
    Settings->Audio.suppressCentroidStrength.setScaleSettings(-2.0f, 2.0f);
}

void GuiManager::setAgentManager(std::shared_ptr<std::vector<AgentManager>> managerList) {
    agentManagerList = managerList;
}

void GuiManager::setAudioManager(std::shared_ptr<AudioManager> manager) {
    audioManager = manager;
}

void GuiManager::setSettings(std::shared_ptr<GLOBAL_SETTINGS> settings) {
    Settings = settings;
}

void GuiManager::setRecursiveShader(std::shared_ptr<sf::Shader> shader) {
    recursiveShader = shader;
}

void GuiManager::setWorldShader(std::shared_ptr<sf::Shader> shader) {
    worldShader = shader;
}