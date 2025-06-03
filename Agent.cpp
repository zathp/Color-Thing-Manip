#include "Agent.hpp"

using namespace std;
using namespace sf;

shared_ptr<GLOBAL_SETTINGS> Agent::GlobalSettings = nullptr;

glm::mat3 Agent::colorMap = glm::mat3(1.0f);

//public members

Agent::Agent(int id) {

	this->id = id;

    //spawn agent with a random position, velocity, and base color
    pos.x = rand() % WIDTH;
    pos.y = rand() % HEIGHT;
    dir = (float)(rand() % 10000) / 10000 * 2 * _Pi;
    vel = Vector2f(cos(dir), sin(dir));

    randomizeColorBase();
}

void Agent::randomizeColorBase() {
	colorBase = Color(
		(float)(rand() % 206 + 50),
		(float)(rand() % 206 + 50),
		(float)(rand() % 206 + 50)
	);
}

void Agent::updateColorBase(Color c, float rate) {
    colorBase = Color(
        colorBase.r + (c.r - colorBase.r) * rate,
        colorBase.g + (c.g - colorBase.g) * rate,
        colorBase.b + (c.b - colorBase.b) * rate
    );
}

void Agent::applyFilter(Vector3f filter) {
	colorFinal.r *= filter.x;
	colorFinal.g *= filter.y;
	colorFinal.b *= filter.z;
}

void Agent::applyMapping(glm::mat3 map) {

	glm::vec3 newColor = map * glm::vec3(
        colorFinal.r,
        colorFinal.g,
        colorFinal.b
    );

	colorFinal.r = newColor.x;
	colorFinal.g = newColor.y;
	colorFinal.b = newColor.z;
}

glm::vec3 rgbToHsv(const glm::vec3& rgb) {
    float r = rgb.r, g = rgb.g, b = rgb.b;
    float maxC = std::max({ r, g, b });
    float minC = std::min({ r, g, b });
    float delta = maxC - minC;

    float h = 0.f, s = 0.f, v = maxC;

    if (delta != 0.f) {
        if (maxC == r)
            h = fmodf((g - b) / delta, 6.f);
        else if (maxC == g)
            h = (b - r) / delta + 2.f;
        else
            h = (r - g) / delta + 4.f;

        h /= 6.f; // Normalize to [0,1]
        if (h < 0.f) h += 1.f;
    }

    if (maxC != 0.f)
        s = delta / maxC;

    return glm::vec3(h, s, v);
}

// HSV → RGB
glm::vec3 hsvToRgb(const glm::vec3& hsv) {
    float h = hsv.x * 6.f; // sector 0–6
    float s = hsv.y;
    float v = hsv.z;

    int i = int(h);
    float f = h - i;
    float p = v * (1.f - s);
    float q = v * (1.f - f * s);
    float t = v * (1.f - (1.f - f) * s);

    switch (i % 6) {
    case 0: return { v, t, p };
    case 1: return { q, v, p };
    case 2: return { p, v, t };
    case 3: return { p, q, v };
    case 4: return { t, p, v };
    case 5: return { v, p, q };
    }
    return { 0, 0, 0 };
}

void Agent::applyMappingHSV(const glm::mat3& hsvMap) {
    glm::vec3 rgb = { colorFinal.r, colorFinal.g, colorFinal.b };
    glm::vec3 hsv = rgbToHsv(rgb);

    glm::vec3 mapped = hsvMap * hsv;

    // Clamp HSV to safe ranges
    mapped.x = fmodf(mapped.x, 1.f); // hue wraps
    if (mapped.x < 0.f) mapped.x += 1.f;
    mapped.y = glm::clamp(mapped.y, 0.f, 1.f);
    mapped.z = glm::clamp(mapped.z, 0.f, 1.f);

    glm::vec3 newRgb = hsvToRgb(mapped);
    colorFinal.r = newRgb.r;
    colorFinal.g = newRgb.g;
    colorFinal.b = newRgb.b;
}

void Agent::applyHueShift(float hueShift) {
    glm::vec3 hsv = rgbToHsv(glm::vec3(colorFinal.r, colorFinal.g, colorFinal.b));
    hsv.x = fmodf(hsv.x + hueShift, 1.0f);
    if (hsv.x < 0.f) hsv.x += 1.0f;

    glm::vec3 rgb = hsvToRgb(hsv);
    colorFinal.r = rgb.r;
    colorFinal.g = rgb.g;
    colorFinal.b = rgb.b;
}

Color Agent::hueShift(Color c, float rads) {

	glm::vec3 hsv = rgbToHsv(glm::vec3(c.r, c.g, c.b));
	hsv.x = fmodf(hsv.x + rads / (2 * _Pi), 1.0f);
	if (hsv.x < 0.f) hsv.x += 1.0f;
	glm::vec3 rgb = hsvToRgb(hsv);
	return Color(
		static_cast<unsigned char>(rgb.r * 255),
		static_cast<unsigned char>(rgb.g * 255),
		static_cast<unsigned char>(rgb.b * 255)
	);
}

void Agent::applyColorTransformations(shared_ptr<GROUP_SETTINGS> Settings, float time) {

	const Vector2 center = { WIDTH / 2, HEIGHT / 2 };

    if (Settings->Oscillation.timeOscillationEnabled) {
        Vector3f globalTimeFilter = {
            cosfRatio(
                time / (Settings->Oscillation.globalPeriodR * 1000)
            ),
            cosfRatio(
                time / (Settings->Oscillation.globalPeriodG * 1000)
            ),
            cosfRatio(
                time / (Settings->Oscillation.globalPeriodB * 1000)
            )
        };
		applyFilter(globalTimeFilter);
    }

    if (Settings->Oscillation.distAlternate) {
		float dist = sqrt(pow(center.x - pos.x, 2) + pow(center.y - pos.y, 2));
		Vector3f distFilter = {
			cosfRatio(
				dist / Settings->Oscillation.distR / cosfRatio(
					time / (Settings->Oscillation.distPeriodR * 1000)
				)
			),
			cosfRatio(
				dist / Settings->Oscillation.distG / cosfRatio(
					time / (Settings->Oscillation.distPeriodG * 1000)
				)
			),
			cosfRatio(
				dist / Settings->Oscillation.distB / cosfRatio(
					time / (Settings->Oscillation.distPeriodB * 1000)
				)
			)
		};
		applyFilter(distFilter);
    }

    if (GlobalSettings->Audio.audioAlternate) {
		applyFilter(
			Vector3f(
                GlobalSettings->Audio.modR,
                GlobalSettings->Audio.modG,
                GlobalSettings->Audio.modB
			)
		);
    }

    if (Settings->Agents.Color.hueRotationEnabled) {
        float radians = Settings->Agents.Color.hueRotationAngle;

        if (Settings->Agents.Color.hueOscillation) {
            float oscillation = sinf(time / (Settings->Agents.Color.hueOscillationPeriod * 1000.f));
            radians += _Pi * oscillation; // oscillates ±π radians
        }

        float hueShift = radians / (2 * _Pi); // convert to normalized hue offset [0–1]
        applyHueShift(hueShift);
    }
}

Vector2f Agent::getPos() { return pos; }

//place color and advance
void Agent::updatePos(shared_ptr<GROUP_SETTINGS> Settings, Image& im) {

    im.setPixel({ static_cast<unsigned int>(pos.x),static_cast<unsigned int>(pos.y) }, colorFinal);

    //if (Settings->Agents.interpolateColorDrop) {

    //    float dx = pos.x - px;
    //    float dy = pos.y - py;

    //    // unwrap X
    //    if (fabsf(dx) > WIDTH * 0.5f) {
    //        if (dx > 0) {
    //            px += WIDTH;  // unwrap across left edge
    //        }
    //        else {
    //            px -= WIDTH;  // unwrap across right edge
    //        }
    //        dx = x - px;
    //    }

    //    // unwrap Y
    //    if (fabsf(dy) > HEIGHT * 0.5f) {
    //        if (dy > 0) {
    //            py += HEIGHT;
    //        }
    //        else {
    //            py -= HEIGHT;
    //        }
    //        dy = y - py;
    //    }
    //}
    //else {
    //    
    //}

    pos.x += vel.x * Settings->Agents.speed;
    pos.y += vel.y * Settings->Agents.speed;

    //wrap agent positions around screen
    while (pos.x < 0)
        pos.x += WIDTH;
    while(pos.x >= WIDTH)
        pos.x -= WIDTH;
    while(pos.y < 0)
        pos.y += HEIGHT;
    while(pos.y >= HEIGHT)
        pos.y -= HEIGHT;
}

void Agent::updatePos(shared_ptr<GROUP_SETTINGS> Settings) {

    pos.x += vel.x * Settings->Agents.speed;
    pos.y += vel.y * Settings->Agents.speed;

    //wrap agent positions around screen
    while (pos.x < 0)
        pos.x += WIDTH;
    while (pos.x >= WIDTH)
        pos.x -= WIDTH;
    while (pos.y < 0)
        pos.y += HEIGHT;
    while (pos.y >= HEIGHT)
        pos.y -= HEIGHT;
}

void Agent::updateDir(shared_ptr<GROUP_SETTINGS> Settings) {

    Vector3f me = norm(Vector3f(colorFinal.r, colorFinal.g, colorFinal.b));
    Vector3f l = norm(avgColors[0]);
    Vector3f f = norm(avgColors[1]);
    Vector3f r = norm(avgColors[2]);

    float lval = compare(me, l) - Settings->Agents.repulsion;
    float rval = compare(me, r) - Settings->Agents.repulsion;
    float mval = compare(me, f) - Settings->Agents.repulsion;

    float bias = (lval - rval) * (1.0f - mval) * 255.0f;

    // smoothing
    bias = Settings->Agents.memoryFactor * lastTurnBias + (1.0f - Settings->Agents.memoryFactor) * bias;
    lastTurnBias = bias;

    // randomness and turn
    float val = ((float)(rand() % 2000) / 1000.0f - 1.0f) * Settings->Agents.randFactor
        + bias * Settings->Agents.biasFactor;

    alterDir(Settings, val * _Pi / 180.0f * Settings->Agents.turnFactor);

    // reset final color
    colorFinal = colorBase;
}

float Agent::getDir() {
    return dir;
}

int wrap(int i, int size) {
    if (i < 0)
        return size + i % size;
    if (i >= size)
        return i % size;
    return i;
}

void Agent::searchImage(shared_ptr<GROUP_SETTINGS> Settings, Image& im, vector<Vector2i>& normPixels) {

    float angles[3] = {
        dir + Settings->Agents.Search.searchAngleOffset + Settings->Agents.Search.searchAngle,
        dir,
        dir - Settings->Agents.Search.searchAngleOffset - Settings->Agents.Search.searchAngle
    };

    avgColors = { Vector3f(0,0,0), Vector3f(0,0,0), Vector3f(0,0,0) };

    if (normPixels.size() <= 0)
        return;

    for (int pt = 0; pt < normPixels.size(); pt++) {

        int ptx = normPixels[pt].x;
        int pty = normPixels[pt].y;

        for (int dir = 0; dir < 3; dir++) {

            int ptransx =
                pos.x
                + ptx * cosf(angles[dir])
                - pty * sinf(angles[dir])
            ;
            int ptransy =
                pos.y
                + ptx * sinf(angles[dir])
                + pty * cosf(angles[dir])
            ;

            ptransx = wrap(ptransx, WIDTH);
            ptransy = wrap(ptransy, HEIGHT);

            Color c = im.getPixel({ static_cast<unsigned int>(ptransx), static_cast<unsigned int>(ptransy) });

            avgColors[dir].x += c.r;
            avgColors[dir].y += c.g;
            avgColors[dir].z += c.b;
        }
    }

    for (int dir = 0; dir < 3; dir++) {
        avgColors[dir].x /= normPixels.size();
        avgColors[dir].y /= normPixels.size();
        avgColors[dir].z /= normPixels.size();
    }
}

//private members

float Agent::cosfRatio(float val) {
    return cosfRatio(val, 0, 1);
}

float Agent::cosfRatio(float val, float offset, float scale) {
    return (cosf(val * 2 * _Pi) + offset) / scale;
}

void Agent::alterDir(shared_ptr<GROUP_SETTINGS> Settings, float delta) {
    dir = dir + delta;
    vel = Vector2f(cos(dir), sin(dir));

    //lock dir to 8 angles
	if (Settings->Agents.lockDir) {
		float angle = dir / (_Pi / Settings->Agents.lockDirCount);
		angle = round(angle);
		dir = angle * (_Pi / Settings->Agents.lockDirCount);
		vel = Vector2f(cos(dir), sin(dir));
	}
}

float Agent::compare(Vector3f a, Vector3f b) {
    const float maxDist = sqrt(2); //maximum difference between two 3d normalized vectors 
    float val = 1 - (
        sqrt(
            pow(a.x - b.x, 2) +
            pow(a.y - b.y, 2) +
            pow(a.z - b.z, 2)
        )
    ) / maxDist;
    return val;
}

Vector3f Agent::norm(Vector3f v) {
    const Vector3f zero(1.0f / sqrt(3), 1.0f / sqrt(3), 1.0f / sqrt(3));

    float mag = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
    
    if (mag == 0) {
        return zero;
    }
    return Vector3f(v.x / mag, v.y / mag, v.z / mag);
}