# Altamont Pass Project

Wind turbines at Altamont Pass kill a lot of golden eagles and other raptors every year. This project is my attempt at a camera system that spots a bird, tracks where it's going, and "brakes" the turbine if the bird is about to fly into the blades.

It uses OpenCV for the video stuff and YOLOv8 for detecting birds.

## What it does

- Detects birds in a video (or webcam feed) using YOLOv8
- Draws a trail behind the bird so you can see its flight path
- Has a red "danger zone" box that stands in for the turbine blades
- Guesses where the bird will be a second from now based on how it's moving
- If the bird is in the zone (or headed for it), the dashboard flashes BRAKE ENGAGED
- Logs the bird's position to a CSV file and plots the flight path when you quit

## How to run it

```bash
cd "Altamont Pass Project"
source .venv/bin/activate
pip install -r requirements.txt

python main.py                  # runs on eagle_flying.mp4
python main.py myvideo.mp4      # runs on any video you give it
python main.py webcam           # runs on your webcam instead
```

Press `q` to quit. Recordings and CSV logs get saved in `recordings/`.

To plot a flight path from the latest CSV:

```bash
python plot_data.py
```

## Files

- `main.py` - the full program
- `plot_data.py` - makes a graph of the flight path from the CSV
- `milestones/` - one script for each day of the build, so you can see how it came together step by step

## Notes

- It uses `best.pt` if there's a custom trained model in the folder. Otherwise it falls back to the regular `yolov8n.pt`, which just detects "bird" in general and not eagles specifically.
- To keep it fast, it only runs YOLO every 5th frame and uses a simpler tracker (KCF) for the frames in between.
- The danger zone is just a fixed box on the screen right now. A real setup would need to map it to where the blades actually are.

## Video credit

`eagle_flying.mp4` is [Kite eagle flying](https://commons.wikimedia.org/wiki/File:Kite_eagle_flying.mp4) from Wikimedia Commons. See that page for the license and author.

## Still to do

- Train a custom model on an eagle dataset from Roboflow
- Test on real footage from Altamont Pass
