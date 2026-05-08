# MotionCanvas: Touchless UI & Computer Vision Drawing Engine

## System Objective
MotionCanvas is a real-time, touchless drawing application that leverages edge AI for spatial tracking. It translates complex 3D hand kinematics into a fluid 2D digital canvas. By engineering a custom gesture-driven state machine, the system allows users to draw, erase, manipulate geometric shapes, and navigate virtual menus entirely through mid-air hand movements—requiring no physical peripherals.

## System Demonstration

*(Click the image below to watch the full demonstration on YouTube)*

<!-- INSTRUCTION: Replace 'YOUR_YOUTUBE_VIDEO_ID' in both the image link and the youtube link below with the actual ID of your video (the string of letters/numbers after '?v=' in the URL). -->

<div align="center">
  <a href="https://www.youtube.com/watch?v=yYzUuUd7BlU">
    <img src="https://img.youtube.com/vi/yYzUuUd7BlU/maxresdefault.jpg" alt="MotionCanvas Demonstration" width="800" style="border-radius: 8px;">
  </a>
</div>

<!-- ALTERNATIVE AUTOPLAY INSTRUCTION: If you decide you want the video to silently autoplay directly on the page instead of linking to YouTube, delete the block above and drag-and-drop your raw .mp4/.mov file right here in the GitHub editor. -->

---

## System Architecture & HCI Logic
The application is built entirely in Python, utilizing OpenCV for rendering and Google's MediaPipe for skeletal tracking.

### 1. Spatial Vision & Kinematics
* **Real-Time Tracking:** Extracts 21 3D hand landmarks per frame. 
* **State-Machine Logic:** The system continuously evaluates the relative Y-coordinates and Euclidean distances between specific joints (e.g., comparing the fingertip to the PIP joint) to determine intent, eliminating false positives during rapid movement.

### 2. Gesture-Driven Controls
* **Drawing Mode:** Triggered when only the index finger is extended. Coordinates are smoothed over time using a custom interpolation factor to remove camera jitter.
* **Eraser Mode:** Triggered when all five fingers are extended (Open Hand). The system calculates a spatial radius around the palm center to clear canvas data.
* **Shape Manipulation:** By pinching the thumb and index finger, the user can spawn and dynamically scale complex polygons (Triangles, Hexagons, Stars, Circles) based on the Euclidean distance between the two fingertips.

### 3. Virtual UI Overlay
* **Contextual Menus:** Activating the menu gesture overlays a semi-transparent UI. The camera feed dynamically blurs (Gaussian Blur) to pull focus to the menu elements.
* **Collision Detection:** Menu selection is handled via custom point-in-polygon calculations, allowing the user's virtual "cursor" to interact with non-rectangular hitboxes like the color wheel.

---

## Quick Start Guide

### Prerequisites
* **Python 3.8+**
* A functional webcam.

### Installation
Clone the repository and install the required dependencies:
```bash
pip install opencv-python mediapipe numpy
```

### Execution
Run the core engine. Ensure that the `canvas_icons`, `menu`, `colorwheel`, `line_thickness`, and `eraser_size` asset folders are located in the same root directory as the script.
```bash
python MotionCanvas.py
```

### Basic Controls
* **Draw:** Extend your Index Finger only.
* **Erase:** Extend all five fingers and wave over the drawing.
* **Place Shape:** Select a shape from the left-hand toolbar. Pinch your Thumb and Index finger together to scale the shape, and release to stamp it onto the canvas.
* **Open Menu:** Extend Middle, Ring, and Pinky fingers. (Allows access to color wheel, line thickness, and save functions).
* **Save Canvas:** Navigate to the save icon in the virtual menu to export the canvas as a high-resolution `.png`.

---
