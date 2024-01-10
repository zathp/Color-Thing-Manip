#include "GUI.hpp"

#include <iostream>
#include <string>

#include <SFGUI/Viewport.hpp>

#pragma warning(disable: 26495)
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)

using namespace std;
using namespace sf;
using namespace sfg;

//-------------------------------------------------------------------------------------
//integer setting, can be optionally attached to the index of a list with given option names
Setting<int>::Setting(int* val, const string& name, vector<string>& map,
	std::function<void()> onChange) : SettingBase(name, onChange) {

	combo = ComboBox::Create();
	combo->SetRequisition(guiOptions.entrySize);

	for (int i = 0; i < map.size(); i++) {
		combo->AppendItem(map[i].substr(0, 18));
	}

	combo->SelectItem(*val);
		
	combo->GetSignal(ComboBox::OnSelect).Connect([val, this]() {
		*val = combo->GetSelectedItem();
	});

	box_main->Pack(combo);
}

Setting<int>::Setting(int* val, string name,
	function<void()> onChange) : SettingBase<int>(name, onChange) {
		
	createEntry(input, val, getString(val));
}

int Setting<int>::parse(const string &s, size_t* idx) {
	int i = stoi(s, idx);
	return i;
}


//-------------------------------------------------------------------------------------
//float setting
Setting<float>::Setting(float* val, std::string name, float initialVal,
	function<void()> onChange) : SettingBase<float>(name, onChange) {

	createEntry(input, val, fString(to_string(initialVal)));
}

Setting<float>::Setting(float* val, string name,
	function<void()> onChange) : SettingBase<float>(name, onChange) {

	createEntry(input, val, fString(getString(val)));
}

float Setting<float>::parse(const string &s, size_t* idx) {
	float f = stof(s, idx);
	return f;
}


//-------------------------------------------------------------------------------------
//boolean setting
Setting<bool>::Setting(bool* val, string name,
	function<void()> onChange) : SettingBase<bool>(name, onChange) {

	createCheckBox(input, val);
}


//-------------------------------------------------------------------------------------
//dropdown widget
DropDown::DropDown(string name) {

	box_main = Box::Create(Box::Orientation::VERTICAL);
	box_main->SetRequisition(guiOptions.itemSize);

	box_label = Box::Create(Box::Orientation::HORIZONTAL);
	box_label->SetRequisition(guiOptions.itemSize);

	box_dropdown_contain = Box::Create(Box::Orientation::HORIZONTAL);
	box_dropdown = Box::Create(Box::Orientation::VERTICAL);

	box_main->Pack(box_label, false, false);
	box_main->Pack(box_dropdown_contain);

	GUIBase::packSpacerBox(box_dropdown_contain, box_pad, guiOptions.leftPaddingSize);
	box_dropdown_contain->Pack(box_dropdown);

	box_dropdown->Show(false);

	btn_expand = Button::Create("+");
	btn_expand->SetClass("dropdown");
	btn_expand->SetRequisition(guiOptions.buttonSize);
	btn_expand->GetSignal(Widget::OnLeftClick).Connect([this]() {
		toggleDropdown();
	});
	box_label->Pack(btn_expand, false, false);

	label = Label::Create(name);
	label->SetClass("group");
	label->SetRequisition(guiOptions.labelSize);
	box_label->Pack(label, false, false);
}

Box::Ptr DropDown::getBase() {

	return box_main;
}

void DropDown::toggleDropdown() {
	if (box_dropdown->IsLocallyVisible())
		Hide();
	else
		Show();
}

void DropDown::addItem(Widget::Ptr item) {

	box_dropdown->Pack(item);
}

void DropDown::addItems(vector<Widget::Ptr> items) {

	for (int i = 0; i < items.size(); i++)
		box_dropdown->Pack(items[i]);
}

void DropDown::Hide() {

	box_dropdown->Show(false);
	btn_expand->SetLabel("+");
	box_main->RequestResize();
}

void DropDown::Show() {

	box_dropdown->Show(true);
	btn_expand->SetLabel("-");
	box_main->RequestResize();
}


//-------------------------------------------------------------------------------------
//gui tab

GUITab::GUITab(const std::string name) : name(name) {
	box_main = Box::Create(Box::Orientation::VERTICAL);
	Hide();
}

void GUITab::addItem(Widget::Ptr item) {
	box_main->Pack(item);
}

void GUITab::Hide() {
	box_main->Show(false);
}

void GUITab::Show() {
	box_main->Show(true);
}


//-------------------------------------------------------------------------------------
//base gui
GUIBase::GUIBase() {

	guiOptions = Options();

	bkgd.setFillColor(Color(128,128,128,128));
	bkgd.setSize(guiOptions.guiSize);

	box_main = Box::Create(Box::Orientation::VERTICAL);

	box_tabBar = Box::Create(Box::Orientation::HORIZONTAL);
	lb_tabBar = Label::Create();
	lb_tabBar->SetRequisition(guiOptions.labelSize);
	btnDec_tabBar = Button::Create("<<");
	btnDec_tabBar->SetRequisition(guiOptions.buttonSize);
	btnInc_tabBar = Button::Create(">>");
	btnInc_tabBar->SetRequisition(guiOptions.buttonSize);

	box_tabBar->Pack(btnDec_tabBar, false, true);
	box_tabBar->Pack(lb_tabBar);
	box_tabBar->Pack(btnInc_tabBar, false, true);

	box_main->Pack(box_tabBar);

	scrollFrame = ScrolledWindow::Create();
	scrollFrame->SetRequisition(guiOptions.guiSize);
	scrollFrame->SetScrollbarPolicy(
		ScrolledWindow::HORIZONTAL_NEVER | ScrolledWindow::VERTICAL_ALWAYS
	);
	box_main->Pack(scrollFrame);

	box_settings = Box::Create(Box::Orientation::VERTICAL);
	scrollFrame->AddWithViewport(box_settings);
}

void GUIBase::addItem(Widget::Ptr item) {
	box_settings->Pack(item);
}

void GUIBase::addItems(vector<Widget::Ptr> items) {

	for (int i = 0; i < items.size(); i++)
		box_settings->Pack(items[i]);
}

void GUIBase::addTab(std::shared_ptr<GUITab> tab) {
	tabs.push_back(tab);
	addItem(tab->box_main);
	if (tabs.size() == 1) {
		setTab(0);
	}
}

void GUIBase::incrementTab() {
	currentTab++;
	if (currentTab >= tabs.size())
		currentTab = 0;
	setTab(currentTab);
}

void GUIBase::decrementTab() {
	currentTab++;
	if (currentTab >= tabs.size())
		currentTab = 0;
	setTab(currentTab);
}

void GUIBase::setTab(int index) {
	for (int t = 0; t < tabs.size(); t++) {
		if (t == index) {
			tabs[t]->Show();
			lb_tabBar->SetText(tabs[t]->name);
		}
		else {
			tabs[t]->Hide();
		}
	}
}

void GUIBase::addToDesktop(Desktop& desktop) {

	desktop.Add(box_main);
}

const RectangleShape GUIBase::getBackground() {

	return bkgd;
}

void GUIBase::Refresh() {

	box_main->RefreshAll();
}

void GUIBase::packSpacerBox(Box::Ptr box_contain, Box::Ptr box_pad, float size) {

	box_pad = Box::Create();
	box_pad->SetRequisition(Vector2f(size, 30));
	box_contain->Pack(box_pad, false, false);

}