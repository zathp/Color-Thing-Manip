#pragma once
#include "Settings.hpp"

using namespace std;
using namespace sfg;
using namespace sf;

class SmallButton : public sfg::Button {
public:
	using Ptr = std::shared_ptr<SmallButton>;

	static Ptr Create(Vector2f size) {
		auto ptr = Ptr(new SmallButton);
		ptr->size = size;
		return ptr;
	}

protected:
	Vector2f size;
	sf::Vector2f CalculateRequisition() override {
		// Hardcode a minimal size
		return size;  // tweak as needed
	}
};

//-------------------------------------------------------------------------------------
//SettingBase

//-------------------------------------------------------------------------------------
//integer setting, can be optionally attached to the index of a list with given option names
Setting<int>::Setting(int val, const string& name, const vector<string>& map,
	std::function<void()> onChange) : SettingBase(name, onChange) {
	this->val = val;
	combo = ComboBox::Create();
	combo->SetRequisition(guiOptions.entrySize);

	updateComboBoxItems(map);

	combo->SelectItem(val);

	combo->GetSignal(ComboBox::OnSelect).Connect([this]() {
		this->val = combo->GetSelectedItem();
		this->onChange();
	});

	combo->GetSignal(ComboBox::OnOpen).Connect([this]() {
		box_container->RequestResize();
		combo->UpdateDrawablePosition();
	});

	box_container->Pack(combo);

	box_main->Pack(verticalSpacer(10), false, false);
}

Setting<int>::Setting(int val, const string& name,
	function<void()> onChange) : SettingBase<int>(name, onChange) {

	this->val = val;

	input = sfg::Entry::Create(getString());
	input->SetClass("base");
	input->SetRequisition(guiOptions.entrySize);

	input->GetSignal(sfg::Widget::OnLostFocus).Connect([this]() {
		int temp = 0;
		if (tryParseEntry(this->input, temp)) {
			set(temp);
		}
	});

	input->GetSignal(sfg::Entry::OnKeyPress).Connect([this]() {
		if (KeyManager::enter) {
			int temp = 0;
			if (tryParseEntry(this->input, temp)) {
				set(temp);
			}
		}
	});

	box_container->Pack(input, false, false);
	box_main->Pack(verticalSpacer(10), false, false);
}

int Setting<int>::parse(const string& s, size_t* idx) {
	int i = stoi(s, idx);
	return i;
}

void Setting<int>::updateComboBoxItems(const vector<string>& map) {

	if (combo == nullptr) return;

	combo->Clear();
	for (int i = 0; i < map.size(); i++) {
		combo->AppendItem(map[i].substr(0, 18));
	}

	if (val >= 0 && val < map.size()) {
		combo->SelectItem(val);
	}
	else {
		combo->SelectItem(0);
		set(0);
	}

	combo->RequestResize();
}

//-------------------------------------------------------------------------------------
//float setting
Setting<float>::Setting(float val, string name,
	function<void()> onChange) : SettingBase<float>(name, onChange) {

	this->val = val;
	
	input = sfg::Entry::Create(fString(getString()));
	input->SetClass("base");
	input->SetRequisition(guiOptions.entrySize);

	input->GetSignal(sfg::Widget::OnLostFocus).Connect([this]() {
		this->tryUpdateValueFromEntry();
	});

	input->GetSignal(sfg::Entry::OnKeyPress).Connect([this]() {
		if (KeyManager::enter) this->tryUpdateValueFromEntry();
	});

	float range = std::max(std::abs(val) * 10.0f, 1.0f);

	scale = sfg::Scale::Create(-range, range, range / 100.0f);
	scale->SetRequisition(guiOptions.sliderSize);
	scale->SetValue(val);

	scale->GetSignal(sfg::Scale::OnMouseLeftPress).Connect([this]() {
		this->dragging = true;
		this->sliderUpdateClock.restart();
	});

	scale->GetSignal(sfg::Scale::OnMouseLeftRelease).Connect([this]() {
		this->dragging = false;
		this->updateValueFromScale();
	});

	scale->GetSignal(sfg::Scale::OnMouseMove).Connect([this]() {
		if (sliderUpdateClock.getElapsedTime().asMilliseconds() >= sliderUpdateInterval) {
			this->updateValueFromScale();
			sliderUpdateClock.restart();
		}
	});

	//modButton = sfg::Button::Create("%");
	//modButton->SetRequisition({25.f, guiOptions.itemHeight});
	//modButton->GetSignal(sfg::Widget::OnLeftClick).Connect([this]() {
	//	this->toggleModifierBox();
	//	sliderUpdateClock.restart();
	//});

	box_container->Pack(input);
	box_container->Pack(scale);
	//box_container->Pack(modButton);

	//modBox = Box::Create(Box::Orientation::VERTICAL);
	//addModButton = Button::Create("New Modifier");
	//addModButton->SetRequisition({150, 20});
	//addModButton->SetClass("small_font");

	//modListFrame = Frame::Create();
	//modListFrame->SetRequisition({ 200, 200 });



	//modListContainer = Box::Create(sfg::Box::Orientation::VERTICAL);
	//modListFrame->Add(modListContainer);

	//shared_ptr<Modifier> mod = make_shared<Modifier>();

	//localMods.push_back(mod);

	//for (shared_ptr<Modifier> mod : localMods) {
	//	modListContainer->Pack(createModifierEntry(mod), false, false);
	//}
	//modBox->Pack(verticalSpacer(10), false, false);
	//modBox->Pack(addModButton, false, false);
	//modBox->Pack(modListFrame, true, true);

	//box_main->Pack(modBox);
	//modBox->Show(false);
	box_main->Pack(verticalSpacer(10), false, false);
}

void Setting<float>::toggleModifierBox() {
	if (modBox->IsLocallyVisible()) {
		modBox->Show(false);
	}
	else {
		modBox->Show(true);
	}
}

sfg::Widget::Ptr Setting<float>::createModifierEntry(shared_ptr<Modifier> mod) {

	auto box_main = sfg::Box::Create(sfg::Box::Orientation::HORIZONTAL);
	auto active = sfg::CheckButton::Create("");
	auto label = sfg::Label::Create(mod->generateName());
	auto orderButtonsContainer = sfg::Box::Create(sfg::Box::Orientation::VERTICAL);
	auto btn_up = SmallButton::Create({15,15});
	auto btn_down = SmallButton::Create({15,15});

	orderButtonsContainer->Pack(btn_up, false, false);
	orderButtonsContainer->Pack(btn_down, false, false);

	box_main->Pack(active, false, false);
	box_main->Pack(label, true, true);
	box_main->Pack(orderButtonsContainer, false, false);

	return box_main;
}

void Setting<float>::tryUpdateValueFromEntry() {
	float temp = 0;
	if (tryParseEntry(this->input, temp)) {
		set(inputFunc(temp));
		scale->SetValue(temp);
	}
}

void Setting<float>::updateValueFromScale() {
	float temp = scale->GetValue();
	set(inputFunc(temp));
	input->SetText(fString(to_string(temp)));
}

void Setting<float>::setScaleSettings(float min, float max, float step) {
	scale->SetRange(min, max);
	scale->SetIncrements(step, step);
}

float Setting<float>::parse(const string& s, size_t* idx) {
	float f = stof(s, idx);
	return f;
}

//-------------------------------------------------------------------------------------
//boolean setting
Setting<bool>::Setting(bool val, string name,
	function<void()> onChange) : SettingBase<bool>(name, onChange) {
	this->val = val;

	input = sfg::CheckButton::Create("");
	input->SetActive(val);
	input->SetRequisition(guiOptions.entrySize);
	input->GetSignal(sfg::CheckButton::OnToggle).Connect([this]() {
		this->val = input->IsActive();
		this->onChange();
	});
	box_container->Pack(input);
	box_main->Pack(verticalSpacer(10), false, false);
}