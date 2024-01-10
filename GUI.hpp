#pragma once

#include <SFML/Graphics.hpp>

#include <SFGUI/SFGUI.hpp>
#include <SFGUI/Widgets.hpp>

#include <SFGUI/Button.hpp>
#include <SFGUI/Label.hpp>
#include <SFGUI/Entry.hpp>
#include <SFGUI/Box.hpp>
#include <SFGUI/ScrolledWindow.hpp>
#include <SFGUI/ComboBox.hpp>

#include <vector>
#include <functional>

static struct Options {

	Options() {

		const float itemHeight = 30;

		guiSize = sf::Vector2f(300, 600);

		entrySize = sf::Vector2f(100, itemHeight);
		buttonSize = sf::Vector2f(itemHeight, itemHeight);
		labelSize = sf::Vector2f(100, itemHeight);
		itemSize = sf::Vector2f(200, itemHeight);
		leftPaddingSize = 30;
	}

	sf::Vector2f guiSize;

	sf::Vector2f textAlignment;

	sf::Vector2f entrySize;
	sf::Vector2f buttonSize;
	sf::Vector2f labelSize;

	sf::Vector2f itemSize;

	float leftPaddingSize;
} guiOptions;

static class KeyManager {
public:
	static inline bool enter = false;

	static void reset() {
		enter = false;
	}
};

//class ModifierVariable {
//
//
//
//};
//
//class Modifier {
//
//	Setting<float> parent;
//
//	float baseVal;
//
//	Modifier();
//
//};

template<typename T> class SettingBase {

public:

	bool modifiable = false;

	T* val = nullptr;

	std::function<void()> onChange = []() {};

	std::shared_ptr<std::function<T(T)>> inputFunc
		= make_shared<std::function<T(T)>>([](T value) { return value; });

	SettingBase(const std::string &name,
		std::function<void()> onChange = []() {}) : name(name), onChange(onChange) {

		box_main = sfg::Box::Create(sfg::Box::Orientation::HORIZONTAL);

		label = sfg::Label::Create(name);
		label->SetRequisition(guiOptions.labelSize);
		label->SetClass("setting");
		box_main->Pack(label, true, false);
	}

	void setInputFunction(std::function<T(T)> func) {
		*inputFunc = func;
	}

	static std::string getString(T* ptr) {
		try {
			return std::to_string(*ptr);
		}
		catch (const std::exception&) {
			return "*";
		}
	}

	operator sfg::Widget::Ptr() {
		return box_main;
	}

	sfg::Box::Ptr getBase() {
		return box_main;
	}

	bool isEntryValid(sfg::Entry::Ptr entry, T* ptr) {
		if (entry->GetText().isEmpty()
			|| entry->GetText() == "")
			return false;
		std::string text = entry->GetText().toAnsiString();
		try {
			size_t idx;
			T v = parse(text, &idx);
			if (idx == text.size()) {
				*ptr = (*inputFunc)(v);
				onChange();
				return true;
			}
			else {
				return false;
			}
		}
		catch (const std::exception&) {
			return false;
		}
	}

	static std::string fString(const std::string &s) {
		size_t pos = s.find('.');
		if (pos == std::string::npos || pos + 4 > s.length()) return s;
		return s.substr(0, pos + 4);
	}

	static void setEntryValid(sfg::Entry::Ptr entry, bool valid) {
		entry->SetClass(valid ? "valid" : "invalid");
	}

	void createEntry(sfg::Entry::Ptr entry, T* ptr, const std::string& def_input) {

		entry = sfg::Entry::Create(def_input);
		entry->SetClass("base");
		entry->SetRequisition(guiOptions.entrySize);
		entry->GetSignal(sfg::Widget::OnLostFocus).Connect([ptr, entry, this]() {
			this->updateVal(entry, ptr);
		});

		entry->GetSignal(sfg::Entry::OnKeyPress).Connect([ptr, entry, this]() {
			if (KeyManager::enter)
				this->updateVal(entry, ptr);
				
		});
		box_main->Pack(entry);

	}

	void createCheckBox(sfg::CheckButton::Ptr cb, bool* ptr) {

		cb = sfg::CheckButton::Create("");
		cb->SetActive(*ptr);
		cb->SetRequisition(guiOptions.entrySize);
		cb->GetSignal(sfg::CheckButton::OnToggle).Connect([ptr, cb, this]() {
			this->updateBoolVal(cb, ptr);
		});
		box_main->Pack(cb);

	}

	virtual T parse(const std::string& s, size_t* idx) {
		throw std::invalid_argument("invalid setting type");
	}

	void updateVal(sfg::Entry::Ptr entry, T* ptr) {

		if (isEntryValid(entry, ptr)) {
			setEntryValid(entry, true);
		}
		else {
			setEntryValid(entry, false);
		}

	}

	void updateBoolVal(sfg::CheckButton::Ptr cb, bool* ptr) {
		*ptr = cb->IsActive();
		onChange();
	}

protected:

	std::string name;

	sfg::Box::Ptr box_main;
	sfg::Label::Ptr label;
};

template <typename T> class Setting : public SettingBase<T> {};

template <> class Setting<int> : public SettingBase<int> {

public:

	Setting(int* val, const std::string& name, std::vector<std::string>& map,
		std::function<void()> onChange = []() {});

	Setting(int* val, std::string name, std::function<void()> onChange = []() {});

	int parse(const std::string& s, size_t* idx) override;

private:

	sfg::Entry::Ptr input;
	sfg::ComboBox::Ptr combo;
};

template <> class Setting<float> : public SettingBase<float> {

public:

	Setting(float* val, std::string name, float initialVal, std::function<void()> onChange = []() {});

	Setting(float* val, std::string name, std::function<void()> onChange = []() {});

	float parse(const std::string& s, size_t* idx) override;

private:

	sfg::Entry::Ptr input;
};

template <> class Setting<bool> : public SettingBase<bool> {

public:

	Setting(bool* val, std::string name, std::function<void()> onChange = []() {});

private:

	sfg::CheckButton::Ptr input;
};

class DropDown {

public:

	DropDown(std::string name);

	sfg::Box::Ptr getBase();

	operator sfg::Widget::Ptr() { return box_main; };

	void toggleDropdown();

	void addItem(sfg::Widget::Ptr item);

	void addItems(std::vector<sfg::Widget::Ptr> items);

	void Hide();

	void Show();

private:

	bool* enter;

	sfg::Box::Ptr box_main;

	sfg::Box::Ptr box_label;

	sfg::Button::Ptr btn_expand;
	sfg::Label::Ptr label;

	sfg::Box::Ptr box_dropdown;
	sfg::Box::Ptr box_dropdown_contain;
	sfg::Box::Ptr box_pad;
};

class GUITab {

public:

	std::string name;

	GUITab(const std::string name);

	operator sfg::Widget::Ptr() { return box_main; }

	void addItem(sfg::Widget::Ptr item);

	void Show();

	void Hide();

	sfg::Box::Ptr box_main;
private:

	

};

class GUIBase {

public:

	GUIBase();

	void addItem(sfg::Widget::Ptr item);

	void addItems(std::vector<sfg::Widget::Ptr> items);

	void addTab(std::shared_ptr<GUITab> tab);

	void incrementTab();

	void decrementTab();

	void setTab(int index);

	void addToDesktop(sfg::Desktop& desktop);

	const sf::RectangleShape getBackground();

	void Refresh();

	static void packSpacerBox(sfg::Box::Ptr box_contain, sfg::Box::Ptr box_pad, float size);
	
private:

	int currentTab = 0;

	sf::RectangleShape bkgd;

	sfg::Box::Ptr box_main;

	sfg::Box::Ptr box_tabBar;
	sfg::Label::Ptr lb_tabBar;
	sfg::Button::Ptr btnInc_tabBar;
	sfg::Button::Ptr btnDec_tabBar;

	sfg::ScrolledWindow::Ptr scrollFrame;

	sfg::Box::Ptr box_settings;

	std::vector<std::shared_ptr<GUITab>> tabs;
};