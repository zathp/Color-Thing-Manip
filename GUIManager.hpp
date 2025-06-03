#pragma once

#include "Config.hpp"

#include <SFML/Graphics.hpp>

#include <memory>
#include <vector>

#include "Parameters.hpp"
#include "AgentManager.hpp"
#include "AudioManager.hpp"

class GuiManager {
    std::shared_ptr<std::vector<AgentManager>> agentManagerList;
    std::shared_ptr<AudioManager> audioManager;
    std::shared_ptr<GLOBAL_SETTINGS> Settings;
    std::shared_ptr<GROUP_SETTINGS> GroupSettings;
    std::shared_ptr<sf::Shader> recursiveShader;
    std::shared_ptr<sf::Shader> worldShader;

public:
    GuiManager();
    ~GuiManager();

    void draw();
    void init();

    void setAgentManager(std::shared_ptr<std::vector<AgentManager>> managerList);
    void setAudioManager(std::shared_ptr<AudioManager> manager);
    void setSettings(std::shared_ptr<GLOBAL_SETTINGS> settings);
    void setRecursiveShader(std::shared_ptr<sf::Shader> shader);
    void setWorldShader(std::shared_ptr<sf::Shader> shader);
};