import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math
import tkinter as tk
from tkinter import filedialog, ttk
import threading  # Added for chat window
import queue      # Added for chat window
import tensorflow as tf  # Added for shape detection
from tensorflow import keras  # Added for shape detection

# --- Shape Detection and Message Setup ---
# Load the pre-trained Keras model
try:
    model = tf.keras.models.load_model('model/shapedetector_model_4b.h5', compile=False)
    # Define the class names your model was trained on
    class_names = ['circle', 'rectangle', 'square', 'triangle']
    img_height = 28
    img_width = 28
    print("Shape detection model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Shape detection will be disabled.")
    model = None

# Define the hidden messages for each shape
hidden_messages = {
    'circle': "All Clear",
    'rectangle': "Hold Position",
    'square': "Request Extraction",
    'triangle': "Alert! Possible Threat",
    'unknown': "The shape is unclear... the message remains hidden."
}

# Queue for communication between the CV loop and the Tkinter loop
message_queue = queue.Queue()

# --- End Shape Detection Setup ---


smoothing_factor = 0.5
select_camera = 0

line_thickness = 8
eraser_size = 22
colorIndex = 0

def roi_from_coords(coords, padding=25):
    """
    Creates a 28x28 BGR image from stroke coordinates for the CNN.
    Modified to handle empty coords and maintain aspect ratio.
    """
    if len(coords) < 2:
        # Return an empty white image if not enough points
        return np.zeros((28, 28, 3), dtype=np.uint8) + 255

    img = np.zeros((480, 640, 1), dtype=np.uint8) + 255  # White canvas (1-channel)
    width = 640
    height = 480
    
    # Extract just (x, y) tuples
    plain_coords = [(int(x), int(y)) for x, y, c in coords]  # Ensure they are integers

    for i in range(len(plain_coords) - 1):
        pt1 = plain_coords[i]
        pt2 = plain_coords[i + 1]
        cv2.line(img, pt1, pt2, color=0, thickness=20)  # Draw black line

    x_coords = [x for x, y in plain_coords]
    y_coords = [y for x, y in plain_coords]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, width)
    y_max = min(y_max + padding, height)

    # Ensure min is less than max
    if x_min >= x_max: x_max = x_min + padding
    if y_min >= y_max: y_max = y_min + padding
    
    roi = img[y_min:y_max, x_min:x_max]

    # Handle cases where ROI is empty or invalid
    if roi.size == 0:
        return np.zeros((28, 28, 3), dtype=np.uint8) + 255  # Return white

    # Make ROI square before resizing to avoid distortion
    h, w = roi.shape[:2]
    if h > w:
        pad_w = (h - w) // 2
        roi = cv2.copyMakeBorder(roi, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
    elif w > h:
        pad_h = (w - h) // 2
        roi = cv2.copyMakeBorder(roi, pad_h, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=255)

    try:
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    except cv2.error:
        return np.zeros((28, 28, 3), dtype=np.uint8) + 255 # Return white on error
    
    # Convert from (28, 28) or (28, 28, 1) to (28, 28, 3) BGR
    if len(roi_resized.shape) == 2:
        roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
    else:  # Already has 1 channel
        roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)

    cv2.imwrite('Image.png', roi_bgr)  # Save for debugging
    return roi_bgr

# --- New Function for Shape Prediction ---
def get_shape_prediction(stroke_coords):
    """
    Takes stroke coordinates, prepares the image, and predicts the shape.
    """
    if model is None:
        print("Model not loaded, skipping prediction.")
        return None, 0
        
    if len(stroke_coords) < 5:  # Need at least a few points
        return None, 0

    # Get the 28x28 BGR image (black drawing on white background)
    img_bgr = roi_from_coords(stroke_coords)

    # Convert BGR (OpenCV default) to RGB (Keras/PIL default)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for model
    img_array = keras.preprocessing.image.img_to_array(img_rgb)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Run prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    confidence = 100 * np.max(score)
    
    # Check confidence threshold
    if confidence < 70:  # You can adjust this threshold
        predicted_class = 'unknown'
    else:
        predicted_class = class_names[np.argmax(score)]
        
    print(f"Detected: {predicted_class} with {confidence:.2f}% confidence.")  # For debugging
    
    return predicted_class, confidence

# --- New Function for Chat Window (runs in a separate thread) ---
def create_chat_window():
    """
    Creates and runs the Tkinter chat window.
    """
    try:
        root = tk.Tk()
        root.title("Shape Messages 🤫")
        root.geometry("400x500")
        root.configure(bg="#2C3E50")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#2C3E50")
        style.configure("TScrollbar", troughcolor="#2C3E50", background="#34495E", bordercolor="#2C3E50", arrowcolor="#ECF0F1")
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        chat_log = tk.Text(main_frame, wrap=tk.WORD, bg="#34495E", fg="#ECF0F1",
                           padx=10, pady=10, font=("Helvetica", 11),
                           borderwidth=0, highlightthickness=0)
        
        scrollbar = ttk.Scrollbar(main_frame, command=chat_log.yview)
        chat_log['yscrollcommand'] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        chat_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Define tags for styling the chat
        chat_log.tag_configure('system', foreground="#1ABC9C", font=("Helvetica", 12, "bold"))
        chat_log.tag_configure('shape', foreground="#3498DB", font=("Helvetica", 11, "bold"))
        chat_log.tag_configure('message', foreground="#ECF0F1", font=("Helvetica", 11))
        chat_log.tag_configure('error', foreground="#E74C3C", font=("Helvetica", 11, "italic"))

        chat_log.insert(tk.END, "System:\n", 'system')
        chat_log.insert(tk.END, "Waiting for you to draw a shape...\n\n", 'message')
        chat_log.config(state=tk.DISABLED)

        def check_queue():
            """
            Polls the message queue and updates the chat log.
            """
            try:
                while not message_queue.empty():
                    shape, confidence = message_queue.get_nowait()
                    
                    if shape:
                        message = hidden_messages.get(shape, hidden_messages['unknown'])
                        
                        chat_log.config(state=tk.NORMAL)
                        chat_log.insert(tk.END, "Detection:\n", 'system')
                        
                        if shape == 'unknown':
                            chat_log.insert(tk.END, f"Shape not recognized (Confidence: {confidence:.2f}%)\n", 'error')
                        else:
                            chat_log.insert(tk.END, f"You drew a {shape}! (Confidence: {confidence:.2f}%)\n", 'shape')
                        
                        chat_log.insert(tk.END, f"{message}\n\n", 'message')
                        chat_log.see(tk.END) # Auto-scroll
                        chat_log.config(state=tk.DISABLED)

            except queue.Empty:
                pass  # No new messages
            finally:
                root.after(200, check_queue)  # Check again in 200ms

        root.after(100, check_queue)  # Start the queue checker
        root.mainloop()

    except Exception as e:
        print(f"Failed to create chat window: {e}")
# --- End new functions ---


def draw_shapes(shapes_points,frame,paintWindow,color,line_thickness):
    for shape in shapes_points:
                for i in range(len(shape) - 1):
                            point1 = shape[i]
                            point2 = shape[i + 1]
                            pt1 = (int(point1[0]*640), int(point1[1]*480))
                            pt2 = (int(point2[0]*640), int(point2[1]*480))
                            cv2.line(frame, pt1, pt2, color, line_thickness)
                            cv2.line(paintWindow, pt1, pt2, color, line_thickness)

def calculate_midpoint_and_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return mid_x, mid_y, distance

def move_shape(shape_points, finger_x, finger_y, resize_factor):
    resized_points = [(x * resize_factor, y * resize_factor) for x, y in shape_points]

    centroid_x = sum(x for x, _ in resized_points) / len(resized_points)
    centroid_y = sum(y for _, y in resized_points) / len(resized_points)

    finger_x_normalized = finger_x / 640
    finger_y_normalized = finger_y / 480
    
    translation_x = finger_x_normalized - centroid_x
    translation_y = finger_y_normalized - centroid_y
    
    moved_centroid_x = centroid_x + translation_x
    moved_centroid_y = centroid_y + translation_y
    moved_points = [(x + moved_centroid_x - centroid_x, y + moved_centroid_y - centroid_y) for x, y in resized_points]

    return moved_points

def menu(image, frame):
    start_y = 0
    end_y = 480
    start_x = 0
    end_x = 640

    alpha = image[:, :, 3] / 255.0

    frame_roi = frame[start_y:end_y, start_x:end_x]
    overlay = (1.0 - alpha[:, :, None]) * frame_roi + alpha[:, :, None] * image[:, :, :3]
    frame[start_y:end_y, start_x:end_x] = overlay

def customize(image1, image2, frame):
    start_y1 = 0
    end_y1 = 240
    start_x1 = 0
    end_x1 = 640

    start_y2 = 240
    end_y2 = 480
    start_x2 = 0
    end_x2 = 640

    alpha1 = image1[:, :, 3] / 255.0
    alpha2 = image2[:, :, 3] / 255.0

    frame_roi1 = frame[start_y1:end_y1, start_x1:end_x1]
    overlay1 = (1.0 - alpha1[:, :, None]) * frame_roi1 + alpha1[:, :, None] * image1[:, :, :3]
    frame[start_y1:end_y1, start_x1:end_x1] = overlay1

    frame_roi2 = frame[start_y2:end_y2, start_x2:end_x2]
    overlay2 = (1.0 - alpha2[:, :, None]) * frame_roi2 + alpha2[:, :, None] * image2[:, :, :3]
    frame[start_y2:end_y2, start_x2:end_x2] = overlay2

def customize_menu_area(given_x, given_y, rectangle_top_left, rectangle_bottom_right, num_parts):
    part_number = 99
    rectangle_width = rectangle_bottom_right[0] - rectangle_top_left[0]
    part_width = rectangle_width / num_parts

    if rectangle_top_left[0] <= given_x <= rectangle_bottom_right[0] and rectangle_top_left[1] <= given_y <= rectangle_bottom_right[1]:
        part_number = int((given_x - rectangle_top_left[0]) // part_width)
    return part_number

def colorwheel(image, frame):
    resized_image = cv2.resize(image, (480, 480))

    start_y = 0
    end_y = 480
    start_x = 80
    end_x = 560

    alpha = resized_image[:, :, 3] / 255.0
    frame[start_y:end_y, start_x:end_x, :] = (1.0 - alpha[:, :, None]) * frame[start_y:end_y, start_x:end_x, :] + alpha[:, :, None] * resized_image[:, :, :3]

def point_in_triangle(point, triangle_vertices):
    result = cv2.pointPolygonTest(triangle_vertices, point, False)
    return result >= 0

def apply_blur(frame, blur_amount):
    return cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)

def export_drawing():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    root.destroy()
    if file_path:
        drawing_only_content = paintWindow
        drawing_only_content = cv2.resize(drawing_only_content, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file_path, drawing_only_content)
    else:
        print("Saving cancelled or no file path provided.")

def smooth_points(points, factor):
    smoothed_points = []
    for stroke in points:
        smoothed_stroke = deque()
        if len(stroke) < 2:
            smoothed_stroke.extend(stroke)
        else:
            for i in range(len(stroke) - 1):
                x_smoothed = int((1 - factor) * stroke[i][0] + factor * stroke[i + 1][0])
                y_smoothed = int((1 - factor) * stroke[i][1] + factor * stroke[i + 1][1])
                if len(stroke[i]) > 2: 
                    color_index = stroke[i][2]
                    smoothed_stroke.append((x_smoothed, y_smoothed, color_index))
            if len(stroke[-1]) > 2: 
                smoothed_stroke.append(stroke[-1])
        smoothed_points.append(smoothed_stroke)
    return smoothed_points

def remove_points_within_radius(points, center, radius):
    for stroke in points:
        points_copy = list(stroke)
        for point in points_copy:
            distance = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            if distance < radius:
                stroke.remove(point)

def draw_canvas(window):
    window[1:60, 40:120, :] = (1.0 - clear_icon_resized[:, :, 3]/255.0)[:, :, None] * window[1:60, 40:120, :] + (clear_icon_resized[:, :, 3]/255.0)[:, :, None] * clear_icon_resized[:, :, :3]
    window[1:60, 135:215, :] = (1.0 - color_icons_resized[0][:, :, 3]/255.0)[:, :, None] * window[1:60, 135:215, :] + (color_icons_resized[0][:, :, 3]/255.0)[:, :, None] * color_icons_resized[0][:, :, :3]
    window[1:60, 230:310, :] = (1.0 - color_icons_resized[1][:, :, 3]/255.0)[:, :, None] * window[1:60, 230:310, :] + (color_icons_resized[1][:, :, 3]/255.0)[:, :, None] * color_icons_resized[1][:, :, :3]
    window[1:60, 325:405, :] = (1.0 - color_icons_resized[2][:, :, 3]/255.0)[:, :, None] * window[1:60, 325:405, :] + (color_icons_resized[2][:, :, 3]/255.0)[:, :, None] * color_icons_resized[2][:, :, :3]
    window[1:60, 420:500, :] = (1.0 - color_icons_resized[3][:, :, 3]/255.0)[:, :, None] * window[1:60, 420:500, :] + (color_icons_resized[3][:, :, 3]/255.0)[:, :, None] * color_icons_resized[3][:, :, :3]
    window[1:60, 515:595, :] = (1.0 - color_icons_resized[4][:, :, 3]/255.0)[:, :, None] * window[1:60, 515:595, :] + (color_icons_resized[4][:, :, 3]/255.0)[:, :, None] * color_icons_resized[4][:, :, :3]
    window[90:455, 5:65, :] = shapes_icons

shapes_icons = cv2.imread('canvas_icons/shapes.png',cv2.IMREAD_UNCHANGED)
shapes_icons = cv2.resize(shapes_icons,(60,365),interpolation=cv2.INTER_AREA)
clear_icon = cv2.imread('canvas_icons/Clear.png', cv2.IMREAD_UNCHANGED)
color_icons = [cv2.imread('canvas_icons/Blue.png', cv2.IMREAD_UNCHANGED), 
cv2.imread('canvas_icons/Green.png', cv2.IMREAD_UNCHANGED), 
cv2.imread('canvas_icons/Red.png', cv2.IMREAD_UNCHANGED), 
cv2.imread('canvas_icons/Orange.png', cv2.IMREAD_UNCHANGED), 
cv2.imread('canvas_icons/Black.png', cv2.IMREAD_UNCHANGED)]
clear_icon_resized = cv2.resize(clear_icon, (80, 59), interpolation=cv2.INTER_AREA)
color_icons_resized = [cv2.resize(icon, (80, 59), interpolation=cv2.INTER_AREA) for icon in color_icons]

menu_img = cv2.imread('menu/menu.png', cv2.IMREAD_UNCHANGED)
menu_img = cv2.resize(menu_img, (640, 480), interpolation=cv2.INTER_AREA)
menu_items = []
for i in range(4):
    menu_items.append(cv2.imread(f'menu/select{i+1}.png',cv2.IMREAD_UNCHANGED))
resized_menu_items = []
for i in range(4):
    resized_menu_items.append(cv2.resize(menu_items[i],(640,480),interpolation=cv2.INTER_AREA))
menu_items = resized_menu_items

color_wheel = cv2.imread('colorwheel/colorwheel.png', cv2.IMREAD_UNCHANGED)
color_wheel_colors = []
for i in range(12):
    color_wheel_colors.append(cv2.imread(f'colorwheel/select{i+1}.png', cv2.IMREAD_UNCHANGED))

line_list = []
for i in range(10):
    line_list.append(cv2.imread(f'line_thickness/line{i+1}.png', cv2.IMREAD_UNCHANGED))
resized_line_list = []
for i in range(10):
    resized_line_list.append(cv2.resize(line_list[i],(640,240),interpolation=cv2.INTER_AREA))
line_list = resized_line_list

eraser_list = []
for i in range(10):
    eraser_list.append(cv2.imread(f'eraser_size/eraser{i+1}.png',cv2.IMREAD_UNCHANGED))
resize_eraser_list = []
for i in range(10):
    resize_eraser_list.append(cv2.resize(eraser_list[i],(640,240),interpolation=cv2.INTER_AREA))
eraser_list = resize_eraser_list


triangle1 = np.array([(300, 106), (370, 106), (335, 240)])
triangle2 = np.array([(371, 106), (432, 141), (335, 240)])
triangle3 = np.array([(433, 142), (468, 202), (335, 240)])
triangle4 = np.array([(469, 203), (469, 274), (335, 240)])
triangle5 = np.array([(470, 275), (432, 335), (335, 240)])
triangle6 = np.array([(433, 336), (370, 370), (335, 240)])
triangle7 = np.array([(371, 371), (300, 371), (335, 240)])
triangle8 = np.array([(301, 372), (229, 335), (335, 240)])
triangle9 = np.array([(229, 335), (204, 274), (335, 240)])
triangle10 = np.array([(205, 275), (203, 203), (335, 240)])
triangle11 = np.array([(204, 204), (238, 142), (335, 240)])
triangle12 = np.array([(238, 153), (299, 105), (335, 240)])
check_triangle = [triangle1,triangle2,triangle3,triangle4,triangle5,triangle6,triangle7,triangle8,triangle9,triangle10,triangle11,triangle12]

shapes = [
    # Square
    [(0.3, 0.3), (0.3, 0.7), (0.3, 0.7), (0.7, 0.7), (0.7, 0.7), (0.7, 0.3), (0.7, 0.3), (0.3, 0.3)],
    # Triangle
    [(0.5, 0.3), (0.7, 0.7), (0.7, 0.7), (0.3, 0.7), (0.3, 0.7), (0.5, 0.3)],
    # Rotated Hexagon
    [(0.7, 0.5),(0.6, 0.7),(0.4, 0.7),(0.3, 0.5),(0.4, 0.3),(0.6, 0.3),(0.7, 0.5)],
    # Star
    [(0.5, 0.2),(0.58, 0.38),(0.77, 0.38),(0.63, 0.52),(0.68, 0.72),(0.5, 0.62),(0.32, 0.72),(0.37, 0.52),(0.23, 0.38),(0.42, 0.38),(0.5, 0.2)],
    # Pentagon
    [(0.5, 0.3),(0.7, 0.45),(0.6, 0.7),(0.4, 0.7),(0.3, 0.45),(0.5, 0.3)]

]

strokes = [deque(maxlen=1024)]
curr_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 165, 255), (0, 0, 0), (35,70,239), (33,150,248), (22,186,253), (26,234,243), (46,218,199), (73,181,69), (209,147,31), (167,90,61), (147,54,78), (150,59,135), (79,30,185), (91,68,226)]

paintWindow = np.zeros((480, 640, 3), dtype=np.uint8) + 255
cv2.namedWindow('MotionCanvas', cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(select_camera)
returning_frame = True

paused = True
no_hands = True
menu_active = False
frame_count = 0
option = 0
exit_program = False
color_selected = False
drawing_var = True
eraser_active = False
shapes_mode = False
resized_points = []
final_points = []
add_new_shape = False

# --- Start the chat window in a separate thread ---
chat_thread = threading.Thread(target=create_chat_window, daemon=True)
chat_thread.start()
# ---

while returning_frame:
    returning_frame, frame = cap.read()
    if not returning_frame:
        break # Exit if frame read fails

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    draw_canvas(frame)

    if colorIndex == 0:
        frame =  cv2.rectangle(frame, (135, 1), (215, 60), (255, 255, 255), 3)
        point_color = (255, 0, 0)
    elif colorIndex == 1:
        frame = cv2.rectangle(frame, (230, 1), (310, 60), (255, 255, 255), 3)
        point_color = (0, 255, 0)
    elif colorIndex == 2:
        frame = cv2.rectangle(frame, (325, 1), (405, 60), (255, 255, 255), 3)
        point_color = (0, 0, 255)
    elif colorIndex == 3:
        frame = cv2.rectangle(frame, (420, 1), (500, 60), (255, 255, 255), 3)
        point_color = (0, 165, 255)
    elif colorIndex == 4:
        frame = cv2.rectangle(frame, (515, 1), (595, 60), (255, 255, 255), 3)
        point_color = (0, 0, 0)

    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            center = (landmarks[8][0], landmarks[8][1])
            thumb = (landmarks[4][0], landmarks[4][1])
            middle_finger = (landmarks[12][0], landmarks[12][1])

            cv2.putText(frame, "Hand Detected", (475, 450), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
            index_finger_straight = landmarks[8][1] < landmarks[6][1]
            middle_finger_straight = landmarks[12][1] < landmarks[10][1]
            ring_finger_straight = landmarks[16][1] < landmarks[14][1]
            small_finger_straight = landmarks[20][1] < landmarks[18][1]
            thumb_straight = landmarks[4][0] < landmarks[2][0]
            thumb_straight_y = landmarks[4][1] < landmarks[2][1]
            if middle_finger_straight and ring_finger_straight and small_finger_straight == False:
                menu_active = True
                frame = apply_blur(frame, blur_amount=15)
                frame_count +=1
                if (thumb[1] - center[1] < 15) and frame_count>=6:
                    if option>=4:
                        option=0
                    else:
                        option+=1
                    frame_count=0
                if option==0:
                    menu(menu_img,frame)
                elif option==1:
                    menu(menu_items[0],frame)
                elif option==2:
                    menu(menu_items[1],frame)
                elif option==3:
                    menu(menu_items[2],frame)
                elif option==4:
                    menu(menu_items[3],frame)
            elif menu_active:
                if option == 0:
                    drawing_var = True
                    menu_active =  False
                if option==4:
                    exit_program = True
                if option==3:
                    export_drawing()
                    option = 0
                    drawing_var = True
                if option==1:
                    frame = apply_blur(frame, blur_amount=15)
                    if (thumb[1] - center[1] <= 20):
                        drawing_var = False
                        for i in range(12):
                            if point_in_triangle(center,check_triangle[i]):
                                colorwheel(color_wheel_colors[i],frame)
                                color_selected = True
                                colorIndex = i+5
                    elif (thumb[1]-center[1]>30) and color_selected == True:
                        color_selected = False
                        option=0
                        drawing_var = True
                        menu_active =  False
                    else:
                        colorwheel(color_wheel,frame)
                if option==2:
                    frame = apply_blur(frame, blur_amount=7)
                    line_value = line_thickness - 1
                    eraser_value = (eraser_size-22)//2
                    customize(eraser_list[eraser_value],line_list[line_value],frame)
                    if (thumb[1] - center[1] < 30):
                        line_value = customize_menu_area(center[0],center[1],(150, 390), (484, 430), 10)
                        if line_value == 99:
                            line_value = line_thickness - 1
                        line_thickness_update = line_value+1
                        line_thickness = line_thickness_update
                        eraser_value = customize_menu_area(center[0],center[1],(150, 150), (484, 190), 10)
                        if eraser_value == 99:
                            eraser_value = (eraser_size-22)//2
                        eraser_size_update = (eraser_value+1)*2 + 20
                        eraser_size = eraser_size_update
                        customize(eraser_list[eraser_value],line_list[line_value],frame)
                    elif (thumb[1] - center[1] > 60):
                        option=0
                        menu_active = False
                        drawing_var = True
                        
            if menu_active == False:
                if index_finger_straight and middle_finger_straight and ring_finger_straight and thumb_straight and small_finger_straight:
                    remove_points_within_radius(strokes, middle_finger, eraser_size)
                    cv2.putText(frame, "Eraser Mode", (475, 470), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.circle(frame, middle_finger, eraser_size, (255, 255, 255), 2)
                    eraser_active = True
                elif (thumb[1] - center[1] < 30):
                    if paused == True:
                        strokes.append(deque(maxlen=512))
                        curr_index += 1
                        paused = False
                        
                        # --- MODIFICATION: Call Shape Prediction ---
                        # Predict the shape from the stroke that was *just* finished
                        if curr_index > 0 and len(strokes[curr_index-1]) > 5: # Check if there's anything to predict
                            try:
                                # Get the stroke data
                                stroke_to_predict = strokes[curr_index-1]
                                # Run prediction
                                shape, confidence = get_shape_prediction(stroke_to_predict)
                                if shape:
                                    # Send the result to the chat window
                                    message_queue.put((shape, confidence))
                            except Exception as e:
                                print(f"Error during prediction: {e}")
                        # --- End Modification ---
                        
                    if eraser_active == False:
                        cv2.putText(frame, "Paused", (475, 470), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                elif center[1] <= 60:
                    if 40 <= center[0] <= 120:
                        strokes = [deque(maxlen=1024)]
                        curr_index = 0
                        paintWindow = np.zeros((480, 640, 3), dtype=np.uint8) + 255
                    elif 135 <= center[0] <= 205:
                        colorIndex = 0
                    elif 230 <= center[0] <= 310:
                        colorIndex = 1
                    elif 325 <= center[0] <= 405:
                        colorIndex = 2
                    elif 420 <= center[0] <= 500:
                        colorIndex = 3
                    elif 515 <= center[0] <= 595:
                        colorIndex = 4
                elif center[0] <=60:
                    if 90 <= center[1] <= 150:
                        selected_shape = shapes[0]
                        shapes_mode = True
                    elif 151 <= center[1] <= 211:
                        selected_shape = "circle"
                        shapes_mode = True
                    elif 212 <= center[1] <= 272:
                        selected_shape = shapes[1]
                        shapes_mode = True
                    elif 273 <= center[1] <= 333:
                        selected_shape = shapes[2]
                        shapes_mode = True
                    elif 334 <= center[1] <= 394:
                        selected_shape = shapes[3]
                        shapes_mode = True
                    elif 395 <= center[1] <= 455:
                        selected_shape = shapes[4]
                        shapes_mode = True
                else:
                    if shapes_mode == False and index_finger_straight:
                        strokes[curr_index].append((center[0], center[1], colorIndex))
                        paused = True
                        no_hands = True
                        cv2.circle(frame, center, 6, colors[colorIndex], -1)
                        cv2.circle(frame, center, 6, (255, 255, 255), 1)
                        eraser_active = False
                if shapes_mode and middle_finger_straight == False:
                    resized_points = selected_shape
                    center_pointx,center_pointy,difference = calculate_midpoint_and_distance(center,thumb)
                    if selected_shape == 'circle':
                        circle_center = (center_pointx,center_pointy)
                        cv2.circle(frame,circle_center,int(difference),colors[colorIndex],line_thickness)
                        theta = np.linspace(0, 2 * np.pi, 360)  # Generate 360 points
                        x_boundary = (circle_center[0] + int(difference) * np.cos(theta)).astype(int)
                        y_boundary = (circle_center[1] + int(difference) * np.sin(theta)).astype(int)
                        # Extract boundary points as a list of (x, y) tuples
                        resized_points = [(x, y, colorIndex) for x, y in zip(x_boundary, y_boundary)]
                    else:
                        resize_factor = difference/120
                        resize_factor = round(resize_factor,1)
                        resized_points = move_shape(resized_points,center_pointx,center_pointy,resize_factor)
                        shape_color = colors[colorIndex]
                        for i in range(len(resized_points) - 1):
                            point1 = resized_points[i]
                            point2 = resized_points[i + 1]
                            color = colors[colorIndex]
                            pt1 = (int(point1[0]*640), int(point1[1]*480))
                            pt2 = (int(point2[0]*640), int(point2[1]*480))
                            cv2.line(frame, pt1, pt2, color, line_thickness)
                            cv2.line(paintWindow, pt1, pt2, color, line_thickness)
                elif middle_finger_straight and shapes_mode:
                    shapes_mode = False
                    add_new_shape = True
    else:
        if no_hands:
            strokes.append(deque(maxlen=512))
            curr_index += 1
            no_hands = False

        cv2.putText(frame, "No Detections", (475, 450), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

    if menu_active == False and drawing_var:
        paintWindow = np.zeros((480, 640, 3), dtype=np.uint8) + 255
        cv2.namedWindow('MotionCanvas', cv2.WINDOW_AUTOSIZE)
        if add_new_shape:
            if selected_shape == 'circle':
                strokes.append(resized_points)
                strokes.append(deque(maxlen=512))
                curr_index +=2
            else:
                for i in range(len(resized_points)):
                    point = resized_points[i]
                    pt1 = (int(point[0]*640),int(point[1]*480),colorIndex)
                    final_points.append(pt1)
                strokes.append(final_points)
                strokes.append(deque(maxlen=512))
                curr_index +=2
            final_points = []
            add_new_shape = False
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                point1 = stroke[i]
                point2 = stroke[i + 1]
                color_index = point1[2]
                color = colors[color_index]
                pt1 = (point1[0], point1[1])
                pt2 = (point2[0], point2[1])
                cv2.line(frame, pt1, pt2, color, line_thickness)
                cv2.line(paintWindow, pt1, pt2, color, line_thickness)
    
    if cv2.waitKey(1) == ord('t'):
         resized_points = shapes[0]
         resize_factor = 0.5
         
    cv2.imshow("WebCam", frame)
    cv2.imshow("MotionCanvas", paintWindow)
        
    if exit_program:
        break
cap.release()
cv2.destroyAllWindows()

# Note: The second part of your provided code (the 'App' class)
# is not needed here, as its functionality (model loading, prediction)
# has been integrated directly into your main MotionCanvas script.