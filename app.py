from flask import Flask, render_template, request, jsonify, session
from PIL import Image
import numpy as np
import os
import cv2  
from rembg import remove
from io import BytesIO
import mediapipe as mp
from skimage.color import rgb2lab
import base64

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Replace with a strong secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function (from your Streamlit code)
def is_surrounded_by_dark(img_lab, x, y, threshold=60, ratio=0.5):
    h, w = img_lab.shape[:2]
    count_dark = 0
    total = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                L = img_lab[ny, nx, 0]
                if L < threshold:
                    count_dark += 1
                total += 1
    return (count_dark / total) >= ratio

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            try:
                image = Image.open(file.stream).convert("RGB")

                # Remove background
                result_no_bg = remove(image)
                image_np = np.array(Image.open(BytesIO(result_no_bg)) if isinstance(result_no_bg, bytes) else result_no_bg)
                if image_np.shape[2] == 3:
                    alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
                    image_np = np.dstack((image_np, alpha))

                # Resize image (similar to Streamlit)
                max_width = 800
                scale = max_width / image_np.shape[1]
                image_np_resized = cv2.resize(image_np, (0, 0), fx=scale, fy=scale)
                h, w = image_np_resized.shape[:2]

                # Mediapipe Pose and Face Mesh
                mp_pose = mp.solutions.pose
                mp_face = mp.solutions.face_mesh

                y_shoulder = h # Default in case pose not found
                with mp_pose.Pose(static_image_mode=True) as pose:
                    result_pose = pose.process(cv2.cvtColor(image_np_resized[:, :, :3], cv2.COLOR_RGB2BGR))
                    if result_pose.pose_landmarks:
                        shoulder_l = result_pose.pose_landmarks.landmark[11]
                        shoulder_r = result_pose.pose_landmarks.landmark[12]
                        y_shoulder = int(min(shoulder_l.y, shoulder_r.y) * h)

                cropped = image_np_resized[:y_shoulder, :, :]

                cx, cy = w // 2, y_shoulder - 50 # Default if face not found
                L_ref = 50 # Default L_ref

                with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
                    result_face = face_mesh.process(cv2.cvtColor(cropped[:, :, :3], cv2.COLOR_RGB2BGR))
                    if result_face.multi_face_landmarks:
                        landmarks = result_face.multi_face_landmarks[0]
                        chin = landmarks.landmark[152]
                        cx, cy = int(chin.x * cropped.shape[1]), int(chin.y * cropped.shape[0])
                        cy_ref = max(0, cy - 20)
                        img_rgb = cropped[:, :, :3]
                        img_lab = rgb2lab(img_rgb)
                        L_ref = np.mean(img_lab[max(cy_ref - 2, 0):cy_ref + 3, max(cx - 2, 0):cx + 3, 0])

                # Store necessary data in session to be used by the slider update
                session['original_cropped_image_np'] = cropped.tolist() # Convert to list for JSON serialization
                session['L_ref'] = L_ref
                session['cx'] = cx
                session['cy'] = cy

                # Initial processing with default slider value
                value = 5 # Default slider value
                final_img_initial = apply_alpha_mask(cropped, L_ref, value, cx, cy)

                # Convert processed image to base64 for display
                img_pil = Image.fromarray(final_img_initial)
                buf = BytesIO()
                img_pil.save(buf, format="PNG")
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                return jsonify({'success': True, 'image': img_base64})

            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('process.html')

@app.route('/update_image', methods=['POST'])
def update_image():
    if 'original_cropped_image_np' not in session:
        return jsonify({'error': 'Image not found in session'}), 400

    cropped_np = np.array(session['original_cropped_image_np'], dtype=np.uint8)
    L_ref = session['L_ref']
    cx = session['cx']
    cy = session['cy']
    slider_value = request.json.get('value', 5) # Get slider value from frontend

    final_img_updated = apply_alpha_mask(cropped_np, L_ref, slider_value, cx, cy)

    img_pil = Image.fromarray(final_img_updated)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({'success': True, 'image': img_base64})

def apply_alpha_mask(cropped_img_np, L_ref, value, cx, cy):
    # This function encapsulates the alpha masking logic from your Streamlit code
    img_rgb = cropped_img_np[:, :, :3]
    img_lab = rgb2lab(img_rgb)
    alpha_mask = cropped_img_np[:, :, 3].copy()
    h_cropped, w_cropped = cropped_img_np.shape[:2]

    for y in range(cy, h_cropped):
        for x in range(w_cropped):
            L = img_lab[y, x, 0]
            if L > L_ref + value and not is_surrounded_by_dark(img_lab, x, y):
                alpha_mask[y, x] = 0

    for y in range(cy, h_cropped):
        for x in list(range(0, 30)) + list(range(w_cropped - 30, w_cropped)):
            if img_lab[y, x, 0] > L_ref + value:
                alpha_mask[y, x] = 0

    shoulder_mask = np.zeros(alpha_mask.shape, dtype=np.uint8)
    # Ensure points are within image bounds
    points = np.array([
        [max(0, cx - 60), min(h_cropped -1 , cy + 30)],
        [max(0, cx - 90), min(h_cropped -1, cy + 90)],
        [min(w_cropped -1, cx + 90), min(h_cropped -1, cy + 90)],
        [min(w_cropped -1, cx + 60), min(h_cropped -1, cy + 30)]
    ])
    cv2.fillPoly(shoulder_mask, [points], 255)

    for y in range(cy, h_cropped):
        for x in range(w_cropped):
            if shoulder_mask[y, x] and img_lab[y, x, 0] > L_ref + value:
                alpha_mask[y, x] = 0

    final_img = cropped_img_np.copy()
    final_img[:, :, 3] = alpha_mask
    return final_img


if __name__ == '__main__':
    app.run(debug=True)