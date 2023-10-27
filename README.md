# Color-Thing
A color flow simulation filled with tons of tweakable parameters to generate many different kinds of patterns.
![image](https://github.com/olsonquinn1/Color-Thing/assets/24209366/27749415-33b0-4e26-b090-e94e5632af06)

## How it Works
Each dot is an agent that lives within the image.
It looks at the surrounding areas and determines the average color to the left, right, and front of it.
Using these averages, it compares them to its own and decides which way to turn.

## Controls
![image](https://github.com/olsonquinn1/Color-Thing/assets/24209366/945afbff-0fe7-408e-bf45-dd6990ea3a55)

- Click and hold right-click to pan the view, and scroll to zoom in/out.
- Press 'r' to reset the view.
- Up and Down arrow keys change the selected setting, left and right alter that setting.
- Hold down control or shift to increase the change rate on the setting by 10x or both for 100x.
- There are multiple setting panels that can be accessed by navigating to the panel name (in yellow) and using left or right.
- The triangles visible under agent options are previews of the agents' search pattern.
- Hide/unhide the GUI with 'h'.

## Hardware Acceleration
You can enable the use of CUDA to perform the agent searches. This makes the program much more performant with larger search areas.

Available if you have an Nvidia 1060 or better.

## Audio Visualization
You can apply a color filter to the agents based on the frequencies from an audio input.
If you need an input device for desktop audio, you can use Windows Stereo Mix or VB Cable to give it desktop audio.

You can get VB Cable [here](https://vb-audio.com/Cable/). To get it working with this program, set "CABLE Input" as the default device.
If you want to listen as well, under recording devices in sound options, enable listen to this device on "Cable Output" and choose your usual audio device.

Then select CABLE in the setting for input device under the audio visualization settings tab.

## Screenshots
![image](https://github.com/olsonquinn1/Color-Thing/assets/24209366/ab219a9e-b127-4143-ab3e-be21c4b40574)
![image](https://github.com/olsonquinn1/Color-Thing/assets/24209366/edaa1a60-b784-4300-9d8e-113a8fdbeed8)
