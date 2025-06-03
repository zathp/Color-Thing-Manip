#include "GUI.hpp"

#pragma warning(disable: 26495)
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)

using namespace std;
using namespace sf;
using namespace sfg;


//-------------------------------------------------------------------------------------
//dropdown widget
DropDown::DropDown(string name) {

	box_main = Box::Create(Box::Orientation::VERTICAL);
	box_main->SetRequisition(guiOptions.itemSize);

	box_label = Box::Create(Box::Orientation::HORIZONTAL);
	box_label->SetRequisition(guiOptions.itemSize);

	box_dropdown_contain = Box::Create(Box::Orientation::HORIZONTAL);
	box_dropdown = Box::Create(Box::Orientation::VERTICAL);

	GUIBase::packSpacerBox(box_dropdown_contain, box_pad, guiOptions.leftPaddingSize);
	box_dropdown_contain->Pack(box_dropdown, true, true);

	box_main->Pack(box_label, false, false);
	box_main->Pack(box_dropdown_contain, true, true);

	box_dropdown->Show(false);

	btn_expand = Button::Create("+");
	btn_expand->SetClass("dropdown");
	btn_expand->SetRequisition(guiOptions.buttonSize);
	
	box_label->Pack(btn_expand, false, false);

	label = Label::Create(name);
	label->SetClass("group");
	label->SetRequisition(guiOptions.labelSize);
	box_label->Pack(label, false, false);

	btn_expand->GetSignal(Widget::OnLeftClick).Connect([this]() {
		toggleDropdown();
	});
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
	box_main->RequestResize();
	box_dropdown->RequestResize();
}

void DropDown::addItems(vector<Widget::Ptr> items) {

	for (int i = 0; i < items.size(); i++)
		addItem(items[i]);
}

void DropDown::Hide() {

	box_dropdown->Show(false);
	btn_expand->SetLabel("+");

	box_main->RequestResize();
	box_dropdown->RequestResize();
	box_main->GetParent()->RequestResize();
}

void DropDown::Show() {

	box_dropdown->Show(true);
	btn_expand->SetLabel("-");

	box_main->RequestResize();
	box_dropdown->RequestResize();
	box_main->GetParent()->RequestResize();
}


//-------------------------------------------------------------------------------------
//gui tab

GUITab::GUITab(const std::string name) : name(name) {
	box_main = Box::Create(Box::Orientation::VERTICAL);
	box_main->SetRequisition({guiOptions.guiSize.x, guiOptions.guiSize.y - guiOptions.itemHeight });
	Hide();
}

void GUITab::Hide() {
	box_main->Show(false);
}

void GUITab::Show() {
	box_main->Show(true);
}

//-------------------------------------------------------------------------------------
//parameters gui

GUIParameters::GUIParameters(const std::string name) : GUITab(name) {

	scrollFrame = ScrolledWindow::Create();
	scrollFrame->SetScrollbarPolicy(
		ScrolledWindow::HORIZONTAL_NEVER | ScrolledWindow::VERTICAL_ALWAYS
	);
	scrollFrame->SetRequisition({ guiOptions.guiSize.x, guiOptions.guiSize.y-guiOptions.itemHeight});


	box_settings = Box::Create(Box::Orientation::VERTICAL);

	box_main->Pack(scrollFrame);
	scrollFrame->AddWithViewport(box_settings);

}

void GUIParameters::addItem(sfg::Widget::Ptr item) {
	box_settings->Pack(item);
	box_main->RequestResize();
	scrollFrame->RequestResize();
}

void GUIParameters::Show() {

	box_main->Show(true);
}

//-------------------------------------------------------------------------------------
//base gui
GUIBase::GUIBase() {

	bkgd.setFillColor(Color(64,64,64,200));
	bkgd.setSize(guiOptions.guiSize);

	box_main = Box::Create(Box::Orientation::VERTICAL);

	box_tabBar = Box::Create(Box::Orientation::HORIZONTAL);
	lb_tabBar = Label::Create();
	lb_tabBar->SetRequisition(guiOptions.labelSize);

	btnDec_tabBar = Button::Create("<<");
	btnDec_tabBar->SetRequisition(guiOptions.buttonSize);
	btnDec_tabBar->GetSignal(Button::OnLeftClick).Connect([this]() {
		this->decrementTab();
	});

	btnInc_tabBar = Button::Create(">>");
	btnInc_tabBar->SetRequisition(guiOptions.buttonSize);
	btnInc_tabBar->GetSignal(Button::OnLeftClick).Connect([this]() {
		this->incrementTab();
	});

	box_tabBar->Pack(btnDec_tabBar, false, false);
	box_tabBar->Pack(lb_tabBar);
	box_tabBar->Pack(btnInc_tabBar, false, false);

	box_main->Pack(box_tabBar, false, false);
}

void GUIBase::addTab(std::shared_ptr<GUITab> tab) {

	tabs.push_back(tab);
	box_main->Pack(tab->getBox());

	if (tabs.size() == 1) {
		setTab(0);
	}
}

void GUIBase::packSpacerBox(Box::Ptr box_contain, Box::Ptr box_pad, float size) {
	box_pad = Box::Create();
	box_pad->SetRequisition(Vector2f(size, guiOptions.itemHeight));
	box_contain->Pack(box_pad, false, false);
}

void GUIBase::incrementTab() {
	currentTab++;
	if (currentTab >= tabs.size())
		currentTab = 0;
	setTab(currentTab);
}

void GUIBase::decrementTab() {
	if (currentTab == 0)
		currentTab = static_cast<int>(tabs.size()) - 1;
	else
		currentTab--;

	setTab(currentTab);
}

void GUIBase::setTab(int index) {
	int showIndex = 0;
	for (int t = 0; t < tabs.size(); t++) {
		tabs[t]->Hide();
		if (t == index) {
			showIndex = t;
		}
	}

	tabs[showIndex]->Show();
	lb_tabBar->SetText(tabs[showIndex]->name);

	box_main->RequestResize();
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

//-------------------------------------------------------------------------------------
//modifier editor

shared_ptr<SETTINGS> GUIModifierEditor::Settings = nullptr;

GUIModifierEditor::GUIModifierEditor(const std::string name) : GUITab(name) {
	using namespace sfg;

	// Top input area
	nameField = Entry::Create();
	createButton = Button::Create("Create");

	auto topRow = Box::Create(Box::Orientation::HORIZONTAL, 5.f);
	topRow->Pack(nameField, true, true);
	topRow->Pack(createButton, false, false);

	// Modifier list
	modifierListBox = Box::Create(Box::Orientation::VERTICAL, 2.f);
	modifierListBox->SetRequisition({ 0.f, 0.f });

	modifierScroll = ScrolledWindow::Create();
	modifierScroll->SetScrollbarPolicy(ScrolledWindow::HORIZONTAL_NEVER | ScrolledWindow::VERTICAL_ALWAYS);
	modifierScroll->SetRequisition({ 300.f, 400.f });
	modifierScroll->AddWithViewport(modifierListBox);

	// Editor area
	editorBox = Box::Create(Box::Orientation::VERTICAL, 5.f);
	editorBox->Show(false);

	// Final layout
	box_main->Pack(topRow, false, false);
	box_main->Pack(Box::Create()); // optional spacer
	box_main->Pack(modifierScroll, false, false);
	box_main->Pack(editorBox, false, false);

	// Create button logic
	createButton->GetSignal(Button::OnLeftClick).Connect([this] {
		std::string name = nameField->GetText();
		if (!name.empty() && !Settings->modifierPool.contains(name)) {
			auto mod = std::make_shared<Modifier>();
			Settings->modifierPool[name] = mod;
			refreshModifierList();
			updateFieldVisibility();
		}
		});

	// Add demo modifiers
	//seedModifiers();

	refreshModifierList();
}

void GUIModifierEditor::seedModifiers() {
	//Settings->modifierPool["SineWave"] = std::make_shared<Modifier>(Modifier{
	//	.name = "SineWave",
	//	.type = Modifier::SourceType::Oscillator,
	//	.operation = Modifier::OperationType::Multiply,
	//	.constant = 0.5f,
	//	.speed = 2.0f
	//	});

	//Settings->modifierPool["BassBoost"] = std::make_shared<Modifier>(Modifier{
	//	.name = "BassBoost",
	//	.type = Modifier::SourceType::AudioEnergy,
	//	.operation = Modifier::OperationType::Add,
	//	.audioChannel = 0,
	//	.constant = 0.3f
	//	});

	//Settings->modifierPool["CentroidShift"] = std::make_shared<Modifier>(Modifier{
	//	.name = "CentroidShift",
	//	.type = Modifier::SourceType::AudioCentroidDrift,
	//	.operation = Modifier::OperationType::Subtract,
	//	.audioChannel = 1,
	//	.constant = 0.2f
	//	});
}

void GUIModifierEditor::refreshModifierList() {
	modifierListBox->RemoveAll();

	//for (const shared_ptr<Modifier>& [name, mod] : Settings->modifierPool) {
	//	auto button = sfg::Button::Create(name);
	//	button->SetRequisition({ 280.f, 0.f }); // ensures full width
	//	button->GetSignal(sfg::Button::OnLeftClick).Connect([this, mod] {
	//		showModifierEditor(mod);
	//		});
	//	modifierListBox->Pack(button, false, false);
	//}

	modifierListBox->Show(true);
}


void GUIModifierEditor::showModifierEditor(std::shared_ptr<Modifier> mod) {
	using namespace sfg;

	selectedModifier = mod;
	editorBox->RemoveAll();
	editorBox->Show(true);

	//auto nameLabel = Label::Create("Editing: " + mod->name);

	operationDropdown = ComboBox::Create();
	operationDropdown->AppendItem("Add");
	operationDropdown->AppendItem("Multiply");
	operationDropdown->AppendItem("Divide");
	operationDropdown->AppendItem("Subtract");
	operationDropdown->SelectItem(static_cast<int>(mod->operation));

	sourceDropdown = ComboBox::Create();
	sourceDropdown->AppendItem("AudioEnergy");
	sourceDropdown->AppendItem("AudioCentroidDrift");
	sourceDropdown->AppendItem("AudioPeakDrift");
	sourceDropdown->AppendItem("Oscillator");
	sourceDropdown->AppendItem("Constant");
	sourceDropdown->SelectItem(static_cast<int>(mod->type));

	constantField = SpinButton::Create(0.f, 10.f, 0.01f);
	constantField->SetValue(mod->constant);

	speedField = SpinButton::Create(0.f, 20.f, 0.1f);
	speedField->SetValue(mod->speed);

	channelSelector = std::make_shared<Setting<int>>(
		selectedModifier->audioChannel,
		"Audio Channel"
	);

	// Update `selectedModifier->audioChannel` when changed
	channelSelector->setOnChange([this]() {
		selectedModifier->audioChannel = channelSelector->get();
		updateFieldVisibility();
		});

	// Hook up changes to apply live
	operationDropdown->GetSignal(ComboBox::OnSelect).Connect([this] {
		selectedModifier->operation = static_cast<Modifier::OperationType>(operationDropdown->GetSelectedItem());
		});

	sourceDropdown->GetSignal(ComboBox::OnSelect).Connect([this] {
		selectedModifier->type = static_cast<Modifier::SourceType>(sourceDropdown->GetSelectedItem());
		updateFieldVisibility(); // optional: show/hide fields based on type
		});

	constantField->GetSignal(SpinButton::OnValueChanged).Connect([this] {
		selectedModifier->constant = static_cast<float>(constantField->GetValue());
		});

	speedField->GetSignal(SpinButton::OnValueChanged).Connect([this] {
		selectedModifier->speed = static_cast<float>(speedField->GetValue());
		});

	// Pack layout
	//editorBox->Pack(nameLabel);
	editorBox->Pack(Label::Create("Operation:"));
	editorBox->Pack(operationDropdown);
	editorBox->Pack(Label::Create("Source Type:"));
	editorBox->Pack(sourceDropdown);

	constantLabel = Label::Create("Constant:");
	editorBox->Pack(constantLabel);

	editorBox->Pack(constantField);
	editorBox->Pack(Label::Create("Speed:"));
	editorBox->Pack(speedField);
	editorBox->Pack(Label::Create("Audio Channel:"));
	editorBox->Pack(channelSelector->getBase());

	updateFieldVisibility();
}

void GUIModifierEditor::updateFieldVisibility() {
	auto type = selectedModifier->type;

	switch (type) {
	case Modifier::SourceType::Oscillator:
		constantLabel->SetText("Amplitude:");
		break;

	case Modifier::SourceType::Constant:
		constantLabel->SetText("Constant Value:");
		break;

	case Modifier::SourceType::AudioEnergy:
	case Modifier::SourceType::AudioCentroidDrift:
	case Modifier::SourceType::AudioPeakDrift:
		constantLabel->SetText("Scale Factor:");
		break;
	}

	bool showSpeed = type == Modifier::SourceType::Oscillator;
	bool showChannel = type == Modifier::SourceType::AudioEnergy ||
		type == Modifier::SourceType::AudioCentroidDrift ||
		type == Modifier::SourceType::AudioPeakDrift;

	speedField->Show(showSpeed);
	channelSelector->getBase()->Show(showChannel);
	editorBox->RequestResize();
}