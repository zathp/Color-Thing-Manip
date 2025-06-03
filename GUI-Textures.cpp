#include "GUI-Textures.hpp"

using namespace std;
using namespace sf;

SearchPreview::SearchPreview() {
    tex.resize({ static_cast<unsigned int>(previewSize.x), static_cast<unsigned int>(previewSize.y) });
    tex_rot.resize({ static_cast<unsigned int>(previewSize.x), static_cast<unsigned int>(previewSize.y) });
}

void SearchPreview::update(SEARCH_SETTINGS settings) {
    vector<ConvexShape> searchPreview = { ConvexShape(), ConvexShape(), ConvexShape() };

    //0: left, 1: front, 2: right
    searchPreview[0].setPointCount(3);
    searchPreview[0].setFillColor(sf::Color(255, 0, 0, 150));
    searchPreview[0].setPoint(0, sf::Vector2f(0, 0) + previewPos);
    searchPreview[0].setPoint(1, sf::Vector2f(
        previewScale * cos(_Pi / -2.0f + settings.searchAngleOffset + settings.searchAngle * 3.0f / 2.0f),
        previewScale * sin(_Pi / -2.0f + settings.searchAngleOffset + settings.searchAngle * 3.0f / 2.0f)
    ) + previewPos);
    searchPreview[0].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f + settings.searchAngleOffset + settings.searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f + settings.searchAngleOffset + settings.searchAngle / 2.0f)
    ) + previewPos);

    searchPreview[1].setPointCount(3);
    searchPreview[1].setFillColor(Color(0, 255, 0, 150));
    searchPreview[1].setPoint(0, Vector2f(0, 0) + previewPos);
    searchPreview[1].setPoint(1, Vector2f(
        previewScale * cos(_Pi / -2.0f + settings.searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f + settings.searchAngle / 2.0f)
    ) + previewPos);
    searchPreview[1].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f - settings.searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f - settings.searchAngle / 2.0f)
    ) + previewPos);

    searchPreview[2].setPointCount(3);
    searchPreview[2].setFillColor(Color(0, 0, 255, 150));
    searchPreview[2].setPoint(0, Vector2f(0, 0) + previewPos);
    searchPreview[2].setPoint(1, Vector2f(
        previewScale * cos(_Pi / -2.0f - settings.searchAngleOffset - settings.searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f - settings.searchAngleOffset - settings.searchAngle / 2.0f)
    ) + previewPos);
    searchPreview[2].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f - settings.searchAngleOffset - settings.searchAngle * 3.0f / 2.0f),
        previewScale * sin(_Pi / -2.0f - settings.searchAngleOffset - settings.searchAngle * 3.0f / 2.0f)
    ) + previewPos);

    tex.clear(sf::Color::Transparent);
    for (int i = 0; i < searchPreview.size(); i++) {
        tex.draw(searchPreview[i]);
    }
    tex.display();

    sf::Sprite sprite(tex.getTexture());
    sprite.setRotation(sf::degrees(180)); // or -90, 180
    sprite.setOrigin({ tex.getSize().x / 2.f, tex.getSize().y / 2.f });
    sprite.setPosition({ tex.getSize().y / 2.f, tex.getSize().x / 2.f }); // new center

    tex_rot.clear(sf::Color::Transparent);
    tex_rot.draw(sprite);
    tex_rot.display();
}

//-------------------------------------------------------------------------------------
//Audio Visualizer
AudioVisualizer::AudioVisualizer() {
    tex.resize({ static_cast<unsigned int>(previewSize.x), static_cast<unsigned int>(previewSize.y) });
    tex_rot.resize({ static_cast<unsigned int>(previewSize.x), static_cast<unsigned int>(previewSize.y) });
}

void AudioVisualizer::update(const float* energies, const float* peaks, const float* centroids, const float* peakDrift, const float* centroidDrift) {
	const float smoothingFactor = 0.15f; // between 0 (frozen) and 1 (instant jump)
	const sf::Color colors[3] = {
		sf::Color(255, 0, 0, 150),   // R
		sf::Color(0, 255, 0, 150),   // G
		sf::Color(0, 0, 255, 150)    // B
	};
    const float freqBarOffset = 120.f; // pixels right of energy bars
    const float freqBarWidth = 8.f;

	tex.clear(sf::Color::Transparent);

	for (int i = 0; i < 3; ++i) {
		// Smooth toward new value
		smoothedEnergies[i] = smoothedEnergies[i] * (1.0f - smoothingFactor) + energies[i] * smoothingFactor;

		float height = smoothedEnergies[i] * scaleFactor;
		float baseX = i * (barWidth + barSpacing);
		float baseY = previewSize.y - height;

		// Bar
		sf::RectangleShape bar;
		bar.setSize({ barWidth, height });
        bar.setPosition({ baseX, baseY });
		bar.setFillColor(colors[i]);
		tex.draw(bar);

        // Frequency bar background
        sf::RectangleShape freqBarBg({ freqBarWidth, previewSize.y });
        freqBarBg.setPosition({ baseX + freqBarOffset, 0.f });
        freqBarBg.setFillColor(sf::Color(20, 20, 20, 100));
        tex.draw(freqBarBg);

        // Peak dot
        float peakY = previewSize.y * (1.0f - peaks[i]);
        sf::CircleShape dot(2.f);
        dot.setOrigin({ 2.f, 2.f });
        dot.setPosition({ baseX + freqBarOffset + freqBarWidth / 2.f, peakY });
        dot.setFillColor(sf::Color::White);
        tex.draw(dot);

        // Centroid line
        float centroidY = previewSize.y * (1.0f - centroids[i]);
        sf::RectangleShape line(sf::Vector2f(freqBarWidth - 2.f, 1.f));
        line.setOrigin({ (freqBarWidth - 2.f) / 2.f, 0.5f });
        line.setPosition({ baseX + freqBarOffset + freqBarWidth / 2.f, centroidY });
        line.setFillColor(sf::Color::Yellow);
        tex.draw(line);

        // Drift bars (thin and adjacent to freq bar)
        const float driftBarWidth = 3.f;
        const float driftSpacing = 1.f;

        // Normalize drift to full height
        float peakDriftHeight = peakDrift[i] * 3 * previewSize.y;
        float centroidDriftHeight = centroidDrift[i] * 3 * previewSize.y;

        float driftBaseX = baseX + freqBarOffset + freqBarWidth;

        // Peak Drift bar (White)
        sf::RectangleShape peakDriftBar({ driftBarWidth, peakDriftHeight });
        peakDriftBar.setPosition({ driftBaseX + driftSpacing, previewSize.y - peakDriftHeight });
        peakDriftBar.setFillColor(sf::Color(255, 255, 255, 180));
        tex.draw(peakDriftBar);

        // Centroid Drift bar (Yellow)
        sf::RectangleShape centroidDriftBar({ driftBarWidth, centroidDriftHeight });
        centroidDriftBar.setPosition({ driftBaseX + driftBarWidth + 2 * driftSpacing, previewSize.y - centroidDriftHeight });
        centroidDriftBar.setFillColor(sf::Color(255, 255, 100, 180));
        tex.draw(centroidDriftBar);
	}
    tex.display();

    sf::Sprite sprite(tex.getTexture());
    sprite.setRotation(sf::degrees(180)); // or -90, 180
    sprite.setOrigin({ tex.getSize().x / 2.f, tex.getSize().y / 2.f });
    sprite.setPosition({ tex.getSize().y / 2.f, tex.getSize().x / 2.f }); // new center

    tex_rot.clear(sf::Color::Transparent);
    tex_rot.draw(sprite);
    tex_rot.display();
}
