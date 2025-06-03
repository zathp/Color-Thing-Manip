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
#include <iostream>
#include <string>

#include "Config.hpp"
#include "Settings.hpp"
#include "GUIConfig.hpp"

class DropDown {

	bool* enter;

	sfg::Box::Ptr box_main;

	sfg::Box::Ptr box_label;

	sfg::Button::Ptr btn_expand;
	sfg::Label::Ptr label;

	sfg::Box::Ptr box_dropdown;
	sfg::Box::Ptr box_dropdown_contain;
	sfg::Box::Ptr box_pad;

public:

	DropDown(std::string name);

	sfg::Box::Ptr getBase();

	operator sfg::Widget::Ptr() { return box_main; };

	void toggleDropdown();

	void addItem(sfg::Widget::Ptr item);

	void addItems(std::vector<sfg::Widget::Ptr> items);

	void Hide();

	void Show();
};

class GUITab {

protected:

	sfg::Box::Ptr box_main; // the contents' root box

public:

	std::string name;

	GUITab(const std::string name);

	operator sfg::Widget::Ptr() { return box_main; }

	sfg::Box::Ptr getBox() { return box_main; }

	virtual void Show();

	void Hide();

};

class GUIBase {

	int currentTab = 0;

	sf::RectangleShape bkgd;

	sfg::Box::Ptr box_main;

	sfg::Box::Ptr box_tabBar;
	sfg::Label::Ptr lb_tabBar;
	sfg::Button::Ptr btnInc_tabBar;
	sfg::Button::Ptr btnDec_tabBar;

	std::vector<std::shared_ptr<GUITab>> tabs;

	void incrementTab();

	void decrementTab();

	void setTab(int index);

public:

	GUIBase();

	void addTab(std::shared_ptr<GUITab> tab);

	void addToDesktop(sfg::Desktop& desktop);

	const sf::RectangleShape getBackground();

	void Refresh();

	static void packSpacerBox(sfg::Box::Ptr box_contain, sfg::Box::Ptr box_pad, float size);
	
};

class GUIParameters : public GUITab {

	sfg::ScrolledWindow::Ptr scrollFrame;
	sfg::Box::Ptr box_settings;

public:

	GUIParameters(const std::string name);

	void addItem(sfg::Widget::Ptr item);

	void Show() override;
};

class GUIModifierEditor : public GUITab {

	static std::shared_ptr<SETTINGS> Settings;

	// UI Elements
	sfg::Entry::Ptr nameField;
	sfg::Button::Ptr createButton;
	sfg::Box::Ptr modifierListBox;
	sfg::ScrolledWindow::Ptr modifierScroll;
	sfg::Box::Ptr editorBox;

	// Active editor widgets
	sfg::ComboBox::Ptr sourceDropdown;
	sfg::ComboBox::Ptr operationDropdown;
	sfg::SpinButton::Ptr constantField;
	sfg::SpinButton::Ptr speedField;

	std::shared_ptr<Setting<int>> channelSelector;

	sfg::Label::Ptr constantLabel;

	std::shared_ptr<Modifier> selectedModifier;

	const std::vector<std::string> channelNames = { "Red", "Green", "Blue" };

public:

	GUIModifierEditor(const std::string name);

	void refreshModifierList();

	void showModifierEditor(std::shared_ptr<Modifier> mod);

	void updateFieldVisibility();

	void seedModifiers();

	static void setSettings(std::shared_ptr<SETTINGS> settings) {
		Settings = settings;
	}
};