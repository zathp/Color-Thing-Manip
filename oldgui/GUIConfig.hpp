#pragma once

#include "Config.hpp"
#include "SFML/Graphics.hpp"

static struct GuiOptions {

	const float itemHeight = 30;

	sf::Vector2f guiSize = sf::Vector2f(350, WINDOW_HEIGHT);

	sf::Vector2f entrySize = sf::Vector2f(50, itemHeight);
	sf::Vector2f buttonSize = sf::Vector2f(itemHeight, itemHeight);
	sf::Vector2f labelSize = sf::Vector2f(100, itemHeight);

	sf::Vector2f itemSize = sf::Vector2f(200, itemHeight);

	sf::Vector2f sliderSize = sf::Vector2f(200, itemHeight);

	sf::Vector2f spacerSize = sf::Vector2f(20, 20);

	float leftPaddingSize = 30;

	bool dirtyGui = true;

} guiOptions;

static class KeyManager {
public:
	static inline bool enter = false;

	static void reset() {
		enter = false;
	}
};