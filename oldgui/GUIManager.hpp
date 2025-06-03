#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFGUI/SFGUI.hpp>
#include <SFGUI/Widgets.hpp>
#include <SFGUI/Image.hpp>

#include <algorithm>
#include <array>
#include <unordered_map>

#include "GUI-Textures.hpp"
#include "Agent.hpp"
#include "GUI.hpp"
#include "GUIConfig.hpp"

#include "Settings.hpp"
#include "AgentManager.hpp"
#include "AudioManager.hpp"

class GUIManager {

	struct GUISettings {

		//settings tab
		std::shared_ptr<GUIParameters> settingsTab;

		//dropdowns
		std::shared_ptr<DropDown> settings;
		std::shared_ptr<DropDown> agentOptions;
		std::shared_ptr<DropDown> searchOptions;
		std::shared_ptr<DropDown> oscillationOptions;
		std::shared_ptr<DropDown> shaderOptions;
		std::shared_ptr<DropDown> colorOptions;
		std::shared_ptr<DropDown> audioOptions;

		sfg::Button::Ptr resetColorButton;

		//modifier tab
		std::shared_ptr<GUIModifierEditor> modifierEditor;
	};

	GUISettings guiSettings;

	std::unique_ptr<sfg::SFGUI> sfgui;
	std::unique_ptr<sfg::Desktop> desktop;
	std::unique_ptr<GUIBase> gui;
	sf::RenderTexture guiTexture;
	sf::Sprite guiSprite;

	std::shared_ptr<SETTINGS> Settings;
	std::shared_ptr<AgentManager> agentManager;

	std::shared_ptr<sf::Shader> recursiveShader;
	std::shared_ptr<sf::Shader> worldShader;

	SearchPreview searchPreview;

	std::shared_ptr<AudioManager> audioManager;
	std::function<void()>* searchMethod = nullptr;

	void setupSettings();
	void setupAgentSettings(std::shared_ptr<GUIParameters>);
	void setupSearchSettings(std::shared_ptr<GUIParameters>);
	void setupOscillationSettings(std::shared_ptr<GUIParameters>);
	void setupShaderSettings(std::shared_ptr<GUIParameters>);
	void setupColorSettings(std::shared_ptr<GUIParameters>);
	void setupAudioSettings(std::shared_ptr<GUIParameters>);

	std::unordered_map<std::string, std::shared_ptr<Modifier>> globalModifierPool;

	bool updateRequired = false;
public:

	bool hideGUI = false;

	GUIManager();
	~GUIManager();
	void setFont(std::shared_ptr<sf::Font> font) {
		desktop->GetEngine().GetResourceManager().SetDefaultFont(font);
	}
	void init();
	void update(float, bool heavy);
	void render(sf::RenderWindow&);
	void cleanup();

	void setRecursiveShader(std::shared_ptr<sf::Shader> shader) {
		this->recursiveShader = shader;
	}
	void setWorldShader(std::shared_ptr<sf::Shader> shader) {
		this->worldShader = shader;
	}
	void setAgentManager(std::shared_ptr<AgentManager> agentManager);
	void setAudioManager(std::shared_ptr<AudioManager> audioManager) {
		this->audioManager = audioManager;
	}
	void setSettings(std::shared_ptr<SETTINGS> settings) {
		this->Settings = settings;
	}

	void setScale(sf::Vector2f scale) {
		guiSprite.setScale(scale);
		guiTexture.setView(sf::View(sf::FloatRect(0, 0, guiOptions.guiSize.x, guiOptions.guiSize.y)));
	}

	void handleEvent(sf::Event& event) {
		desktop->HandleEvent(event);
	}

	sf::Sprite getSprite() {
		return guiSprite;
	}

};