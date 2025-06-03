#include "GUIManager.hpp"

using namespace std;
using namespace sf;
using namespace sfg;

GUIManager::GUIManager() {
	// Initialize the GUI
	sfgui = std::make_unique<SFGUI>();
	desktop = std::make_unique<Desktop>();
	desktop->GetEngine().LoadThemeFromFile("style.theme");
}

GUIManager::~GUIManager() {
	cleanup();
}

void GUIManager::init() {
	// Initialize the GUI components
	gui = std::make_unique<GUIBase>();
	gui->addToDesktop(*desktop);
	gui->Refresh();
	guiTexture.create(guiOptions.guiSize.x, guiOptions.guiSize.y);
	guiSprite.setTexture(guiTexture.getTexture());
	guiSprite.setPosition(0, 0);

    setupSettings();
	searchPreview.update(Settings->Agents.Search);
}

void GUIManager::update(float deltaTimeSeconds, bool heavy) {
	// Update the GUI

    desktop->Update(deltaTimeSeconds);
    
    if (guiOptions.dirtyGui || heavy) {
        gui->Refresh();
        guiOptions.dirtyGui = false;
    }

    guiTexture.clear(sf::Color::Transparent);
    guiTexture.draw(gui->getBackground());
    sfgui->Display(guiTexture);
    guiTexture.display();
}

void GUIManager::render(sf::RenderWindow& window) {
	// Render the GUI
	window.draw(guiSprite);
}

void GUIManager::cleanup() {
	// Cleanup the GUI
	gui.reset();
	desktop.reset();
	sfgui.reset();
}

void GUIManager::setAgentManager(std::shared_ptr<AgentManager> agentManager) {
	// Set the agent manager
	this->agentManager = agentManager;
}
void GUIManager::setupSettings() {
	// Setup the settings tab
	guiSettings.settingsTab = make_shared<GUIParameters>("Parameters");
	gui->addTab(guiSettings.settingsTab);

	guiSettings.settings = std::make_shared<DropDown>("Settings");
	guiSettings.settingsTab->addItem(*guiSettings.settings);
	guiSettings.settings->addItem(Settings->targetFPS);
    guiSettings.settings->addItem(Settings->targetUPS);

    setupAgentSettings(guiSettings.settingsTab);
	setupSearchSettings(guiSettings.settingsTab);
	setupOscillationSettings(guiSettings.settingsTab);
	setupShaderSettings(guiSettings.settingsTab);
	setupColorSettings(guiSettings.settingsTab);
	setupAudioSettings(guiSettings.settingsTab);

    guiSettings.modifierEditor = std::make_shared<GUIModifierEditor>("Modifiers");
	gui->addTab(guiSettings.modifierEditor);
}

void GUIManager::setupAgentSettings(std::shared_ptr<GUIParameters> tab) {
    guiSettings.agentOptions = std::make_shared<DropDown>("Agent");
    tab->addItem(*guiSettings.agentOptions);

    Settings->Agents.agentCount.setOnChange([this] {

		if (this->Settings->Agents.agentCount.get() < 1) {
			this->Settings->Agents.agentCount.set(1);
			this->Settings->Agents.agentCount.setEntryText(1);
		}

		if (this->Settings->Agents.agentCount.get() > MAX_AGENTS) {
            this->Settings->Agents.agentCount.set(MAX_AGENTS);
			this->Settings->Agents.agentCount.setEntryText(MAX_AGENTS);
		}

        this->agentManager->updateAgentCount();
    });

	Settings->Agents.timeScale.setScaleSettings(0.0f, 2.0f, 0.01f);
	Settings->Agents.repulsion.setScaleSettings(-1.0f, 1.0f, 0.01f);
	Settings->Agents.memoryFactor.setScaleSettings(-2.0f, 2.0f, 0.01f);

    guiSettings.agentOptions->addItem(Settings->Agents.agentCount);
    guiSettings.agentOptions->addItem(Settings->Agents.speed);
	guiSettings.agentOptions->addItem(Settings->Agents.timeScale);
    guiSettings.agentOptions->addItem(Settings->Agents.turnFactor);
    guiSettings.agentOptions->addItem(Settings->Agents.randFactor);
    guiSettings.agentOptions->addItem(Settings->Agents.biasFactor);
    guiSettings.agentOptions->addItem(Settings->Agents.repulsion);
    guiSettings.agentOptions->addItem(Settings->Agents.lockDir);
    guiSettings.agentOptions->addItem(Settings->Agents.lockDirCount);

	guiSettings.agentOptions->addItem(Settings->Agents.memoryFactor);
    guiSettings.agentOptions->addItem(Settings->Agents.interpolateColorDrop);
}

void GUIManager::setupSearchSettings(std::shared_ptr<GUIParameters> tab) {
    guiSettings.searchOptions = std::make_shared<DropDown>("Search Pattern");
    tab->addItem(*guiSettings.searchOptions);
    guiSettings.searchOptions->addItem(searchPreview.getSfgImage());

    Settings->Agents.Search.searchSize.setOnChange([&] {
        agentManager->updateSearchArea();
    });

    Settings->Agents.Search.searchAngle.setInputFunction([](float f) { return _Pi * f / 180.0f; });
	Settings->Agents.Search.searchAngle.setScaleSettings(0.0f, 180.0f, 1.0f);
    Settings->Agents.Search.searchAngle.setOnChange([&] {
        agentManager->updateSearchArea();
        searchPreview.update(Settings->Agents.Search);
    });

    Settings->Agents.Search.searchAngleOffset.setInputFunction([](float f) { return _Pi * f / 180.0f; });
	Settings->Agents.Search.searchAngleOffset.setScaleSettings(-180.0f, 180.0f, 1.0f);
    Settings->Agents.Search.searchAngleOffset.setOnChange([&] {
        agentManager->updateSearchArea();
        searchPreview.update(Settings->Agents.Search);
    });

	Settings->Agents.Search.searchMethodIndex.updateComboBoxItems(agentManager->searchMethodNames);
    Settings->Agents.Search.searchMethodIndex.setOnChange([&] {
        agentManager->updateSearchArea();
        searchPreview.update(Settings->Agents.Search);
    });

    guiSettings.searchOptions->addItem(Settings->Agents.Search.searchSize);
    guiSettings.searchOptions->addItem(Settings->Agents.Search.searchAngle);
    guiSettings.searchOptions->addItem(Settings->Agents.Search.searchAngleOffset);
    guiSettings.searchOptions->addItem(Settings->Agents.Search.searchMethodIndex);
    guiSettings.searchOptions->addItem(Settings->Agents.Search.useCuda);
}

void GUIManager::setupOscillationSettings(shared_ptr<GUIParameters> tab) {
    guiSettings.oscillationOptions = std::make_shared<DropDown>("Oscillation");
    tab->addItem(*guiSettings.oscillationOptions);

    guiSettings.oscillationOptions->addItem(Settings->Oscillation.timeOscillationEnabled);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.globalPeriodR);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.globalPeriodG);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.globalPeriodB);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distAlternate);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distR);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distG);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distB);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distPeriodR);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distPeriodG);
    guiSettings.oscillationOptions->addItem(Settings->Oscillation.distPeriodB);
}

void GUIManager::setupShaderSettings(std::shared_ptr<GUIParameters> tab) {
    guiSettings.shaderOptions = std::make_shared<DropDown>("Shader");
    tab->addItem(*guiSettings.shaderOptions);

    Settings->Shader.dimRate.setInputFunction([](float f) { return f / 100.0f; });
    Settings->Shader.dimRate.setOnChange([&] {
        recursiveShader->setUniform("dimRate", Settings->Shader.dimRate.get());
    });
	Settings->Shader.dimRate.setScaleSettings(-5.0f, 5.0f, .01f);

    Settings->Shader.disperseFactor.setOnChange([&] {
        recursiveShader->setUniform("disperseFactor", Settings->Shader.disperseFactor.get());
    });
	Settings->Shader.disperseFactor.setScaleSettings(0.0f, 2.0f, .01f);

    guiSettings.shaderOptions->addItem(Settings->Shader.dimRate);
    guiSettings.shaderOptions->addItem(Settings->Shader.disperseFactor);
}

void GUIManager::setupColorSettings(std::shared_ptr<GUIParameters> tab) {
    guiSettings.colorOptions = std::make_shared<DropDown>("Color Filter");
    tab->addItem(*guiSettings.colorOptions);

    guiSettings.resetColorButton = Button::Create("Reset Color");
    guiSettings.resetColorButton->GetSignal(Button::OnLeftClick).Connect([&] {
        agentManager->resetColors();
    });
    guiSettings.colorOptions->addItem(guiSettings.resetColorButton);

    guiSettings.colorOptions->addItem(Settings->Agents.Color.filterR);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.filterG);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.filterB);

    Settings->Agents.Color.colorMapIndex.setOnChange([&] {
        Agent::setColorMap(agentManager->getColorMap());
        worldShader->setUniform("colorMatrix", agentManager->getColorMapGlsl());
    });

    Settings->Agents.Color.colorMapIndex.updateComboBoxItems(agentManager->colorMapNames);

    Settings->Agents.Color.useColorMap.setOnChange([&] {
        worldShader->setUniform("colorMapEnabled",
            Settings->Agents.Color.useColorMap.get() && !Settings->Agents.Color.visibleToAgents.get());
    });

    Settings->Agents.Color.visibleToAgents.setOnChange([&] {
        worldShader->setUniform("colorMapEnabled",
            Settings->Agents.Color.useColorMap.get() && !Settings->Agents.Color.visibleToAgents.get());
    });

    Settings->Agents.Color.hueRotationAngle.setInputFunction([](float f) { return f * 180.0f / _Pi; });

    guiSettings.colorOptions->addItem(Settings->Agents.Color.colorMapIndex);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.useColorMap);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.visibleToAgents);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.hueRotationEnabled);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.hueRotationAngle);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.hueOscillation);
    guiSettings.colorOptions->addItem(Settings->Agents.Color.hueOscillationPeriod);
}

void GUIManager::setupAudioSettings(std::shared_ptr<GUIParameters> tab) {
    guiSettings.audioOptions = std::make_shared<DropDown>("Audio");
    tab->addItem(*guiSettings.audioOptions);

    guiSettings.audioOptions->addItem(audioManager->getVisualizerImage());

	Settings->Audio.deviceIndex.updateComboBoxItems(audioManager->getDeviceNames());
    Settings->Audio.deviceIndex.setOnChange([&] {
        audioManager->setDeviceIndex(Settings->Audio.deviceIndex.get(), true);
    });

    Settings->Audio.audioAlternate.setOnChange([&] {
        audioManager->enableAudio(Settings->Audio.audioAlternate.get());
    });

	Settings->Audio.suppressPeakStrength.setScaleSettings(-2.0f, 2.0f, 0.01f);
	Settings->Audio.suppressCentroidStrength.setScaleSettings(-2.0f, 2.0f, 0.01f);

    guiSettings.audioOptions->addItem(Settings->Audio.deviceIndex);
    guiSettings.audioOptions->addItem(Settings->Audio.audioAlternate);
    guiSettings.audioOptions->addItem(Settings->Audio.autoGainEqualizer);
    guiSettings.audioOptions->addItem(Settings->Audio.gainR);
    guiSettings.audioOptions->addItem(Settings->Audio.gainG);
    guiSettings.audioOptions->addItem(Settings->Audio.gainB);
    guiSettings.audioOptions->addItem(Settings->Audio.peakDriftEnabled);
    guiSettings.audioOptions->addItem(Settings->Audio.peakDriftStrength);
    guiSettings.audioOptions->addItem(Settings->Audio.centroidDriftEnabled);
    guiSettings.audioOptions->addItem(Settings->Audio.centroidDriftStrength);
	guiSettings.audioOptions->addItem(Settings->Audio.suppressLowPeakDrift);
	guiSettings.audioOptions->addItem(Settings->Audio.suppressPeakStrength);
	guiSettings.audioOptions->addItem(Settings->Audio.suppressLowCentroidDrift);
	guiSettings.audioOptions->addItem(Settings->Audio.suppressCentroidStrength);
}


