#include <string>
#include <sstream>
#include <unordered_map>
#include <iomanip>

#include <SFML/Graphics.hpp>

class ProfilerEntry {
public:
    std::string name;

    float current = 0.0f;      // last frame's duration (ms)
    float max = 0.0f;          // max over window
    float avg = 0.0f;          // smoothed average
    float smoothing = 0.9f;    // weight for EMA (exponential moving average)

    sf::Clock timer;

    ProfilerEntry(const std::string& name_, float smoothing_ = 0.9f)
        : name(name_), smoothing(smoothing_) {
    }

    ProfilerEntry() {
        name = "ProfilerEntry";
        smoothing = 0.9f;
    }

    void start() {
        timer.restart();
    }

    void stop() {
        current = timer.getElapsedTime().asMicroseconds() / 1000.0f;
        max = std::max(max, current);
        avg = smoothing * avg + (1.0f - smoothing) * current;
    }

    void resetMax() {
        max = current;
    }
};

class Profiler {
public:
    std::unordered_map<std::string, ProfilerEntry> entries;

    void start(const std::string& name) {
        if (entries.find(name) == entries.end()) {
            entries.emplace(name, ProfilerEntry(name));
        }
        entries[name].start();
    }

    void stop(const std::string& name) {
        entries[name].stop();
    }

    void resetMaxAll() {
        for (auto& [_, entry] : entries) {
            entry.resetMax();
        }
    }

    std::string getReport() const {
        std::stringstream ss;

        // Find longest name length
        size_t nameWidth = 4; // "Name"
        for (const auto& [_, entry] : entries) {
            nameWidth = std::max(nameWidth, entry.name.length());
        }
        nameWidth += 2; // extra padding

        // Header
        ss << std::left << std::setw(nameWidth) << "Name"
            << std::right << std::setw(10) << "Curr(ms)"
            << std::setw(10) << "Avg(ms)"
            << std::setw(10) << "Max(ms)" << "\n";

        ss << std::string(nameWidth + 30, '-') << "\n";

        // Entries
        for (const auto& [_, entry] : entries) {
            ss << std::left << std::setw(nameWidth) << entry.name
                << std::right << std::setw(10) << std::fixed << std::setprecision(2) << entry.current
                << std::setw(10) << std::fixed << std::setprecision(2) << entry.avg
                << std::setw(10) << std::fixed << std::setprecision(2) << entry.max
                << "\n";
        }

        return ss.str();
    }


    Profiler() {}
};

class AutoProfiler {
    Profiler& profiler;
    std::string name;

public:
    AutoProfiler(Profiler& profiler_, const std::string& name_)
        : profiler(profiler_), name(name_) {
        profiler.start(name);
    }

    ~AutoProfiler() {
        profiler.stop(name);
    }
};
