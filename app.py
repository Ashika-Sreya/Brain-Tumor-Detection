import os
import sys
import io
import cv2
import time
import shutil
import threading
import numpy as np
import contextlib
import webbrowser
import subprocess
import platform
import warnings
import tempfile
import requests
import socket  # Import socket for dynamic IP discovery
import datetime

# FastAPI and ReportLab imports
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTasks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

# --- TensorFlow Import and Global Suppression ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logging (INFO, WARNING, ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid specific warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage to avoid GPU-related issues
warnings.filterwarnings("ignore", category=UserWarning, module='keras')  # Suppress Keras/TF warnings

import tensorflow as tf

# Suppress urllib3 warnings (safe to suppress if not verifying SSL, or for local-only use)
from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


@contextlib.contextmanager
def suppress_output():
    """A context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout  # Restore original stdout
            sys.stderr = old_stderr  # Restore original stderr


# Get the base directory of the current script (where app.py resides)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# === APP PORT CONFIGURATION ===
# This port will be used for both local and network access.
APP_PORT = 8000
# ==============================

# --- TensorFlow Model Loading with Suppressed Output ---
model = None  # Initialize model to None globally
model_dir = os.path.join(base_dir, "model")
model_path = os.path.join(model_dir, "brain_tumor_model.keras")

model_load_error_message = None
model_load_success_message = None

if not os.path.exists(model_dir):
    model_load_error_message = f"CRITICAL ERROR: 'model' directory not found at {model_dir}. Please ensure the 'model' folder is in the same directory as app.py."
elif not os.path.exists(model_path):
    model_load_error_message = f"CRITICAL ERROR: Model file 'brain_tumor_model.keras' not found at {model_path}. Please ensure the model file is inside the 'model' folder."
else:
    try:
        with suppress_output():  # Suppress output during model loading
            model = tf.keras.models.load_model(model_path)
        model_load_success_message = f"Model loaded successfully from {model_path}"
    except Exception as e:
        model_load_error_message = f"CRITICAL ERROR: Could not load model from {model_path}: {e}"
        model = None

if model_load_error_message:
    print(model_load_error_message)
elif model_load_success_message:
    print(model_load_success_message)

# Directory configurations
UPLOAD_DIR_NAME = "static/uploads"
GRADCAM_DIR_NAME = "static/gradcam"
IMG_SIZE = (224, 224)
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png'}
last_conv_layer = "Conv_1"  # Adjust if your model's last conv layer has a different name
class_indices = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
label_map = {v: k for k, v in class_indices.items()}

MAX_SAVED_IMAGE_WIDTH_PX = 800
PDF_IMAGE_DISPLAY_WIDTH = 5 / 2.54 * inch
PDF_IMAGE_DISPLAY_HEIGHT = 7 / 2.54 * inch

# Construct full absolute paths for directories
full_upload_dir = os.path.join(base_dir, UPLOAD_DIR_NAME)
full_gradcam_dir = os.path.join(base_dir, GRADCAM_DIR_NAME)
full_static_dir = os.path.join(base_dir, "static")
full_templates_dir = os.path.join(base_dir, "templates")

# Create directories if they don't exist
os.makedirs(full_upload_dir, exist_ok=True)
os.makedirs(full_gradcam_dir, exist_ok=True)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory=full_static_dir), name="static")
app.mount("/gradcam", StaticFiles(directory=full_gradcam_dir), name="gradcam")
templates = Jinja2Templates(directory=full_templates_dir)

# Define custom ReportLab styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Centered', alignment=TA_CENTER))
styles.add(ParagraphStyle(name='ResultText', fontSize=10, leading=12, alignment=TA_LEFT, fontName='Helvetica'))
styles.add(ParagraphStyle(name='DiagnosisRed', fontSize=14, leading=16, alignment=TA_CENTER, textColor='#DC3545', fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='DiagnosisGreen', fontSize=14, leading=16, alignment=TA_CENTER, textColor='#28A745', fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='ImageHeading', fontSize=11, leading=13, alignment=TA_LEFT, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='SmallItalic', fontSize=9, leading=10, alignment=TA_LEFT, fontName='Helvetica-Oblique'))
styles.add(ParagraphStyle(name='DescriptionText', fontSize=10, leading=12, alignment=TA_LEFT, fontName='Helvetica', spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name='TreatmentAdvice', fontSize=10, leading=12, alignment=TA_LEFT, fontName='Helvetica', spaceBefore=6, spaceAfter=6))

# Define descriptions for each tumor type for the PDF report
tumor_descriptions = {
    'no_tumor': "This category indicates that no brain tumor was detected in the provided MRI scan. This is a positive finding, suggesting the absence of a visible mass consistent with a brain tumor.",
    'glioma_tumor': "Gliomas are a common type of brain tumor that originate from glial cells (supportive cells) of the brain or spinal cell. They can vary widely in their aggressiveness and are graded based on their cellular characteristics.",
    'meningioma_tumor': "Meningiomas are typically benign (non-cancerous) tumors that arise from the meninges, the membranes that surround the brain and spinal cell. They are usually slow-growing and can often be monitored or surgically removed.",
    'pituitary_tumor': "Pituitary tumors are growths that develop in the pituitary gland, a small gland located at the base of the brain. Most pituitary tumors are non-cancerous (benign) and can cause symptoms by affecting hormone production or by pressing on nearby brain structures."
}

# Define treatment advice for the PDF report
treatment_advice = {
    'no_tumor': "<b>Recommendation:</b> No tumor was detected in this scan. Continue with regular medical check-ups for ongoing health monitoring. This result is based on an automated system and should always be confirmed by a medical professional.",
    'glioma_tumor': "<b>Recommendation:</b> A glioma tumor was detected. Treatment typically involves a combination of surgery, radiation therapy, and/or chemotherapy, depending on the tumor's grade and location. Immediate consultation with a neuro-oncologist and neurosurgeon is highly recommended for diagnosis confirmation and a personalized treatment plan.",
    'meningioma_tumor': "<b>Recommendation:</b> A meningioma tumor was detected. Treatment options vary based on the tumor's size, location, and symptoms, ranging from watchful waiting and regular monitoring to surgery or radiation therapy. It is advisable to consult with a neurosurgeon or neuro-oncologist for a comprehensive evaluation and treatment strategy.",
    'pituitary_tumor': "<b>Recommendation:</b> A pituitary tumor was detected. Treatment may include medication to control hormone levels, surgical removal of the tumor, or radiation therapy. Consultation with an endocrinologist and neurosurgeon is crucial to determine the most appropriate course of action."
}

# === Grad-CAM Function ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    if model is None:
        raise ValueError("Model is not loaded.")

    if last_conv_layer_name not in [layer.name for layer in model.layers]:
        try:
            conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
            if not conv_layers:
                raise ValueError("No Conv2D layers found in the model for Grad-CAM.")
            last_conv_layer_name = conv_layers[-1].name
        except IndexError:
            raise ValueError("No Conv2D layers found in the model for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[0] == 0:
            raise ValueError("Model predictions are empty.")

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.squeeze(conv_outputs @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def annotate_and_save(orig_img_bgr, label, emoji, filename):
    """
    Annotates the original image with prediction and saves it.
    """
    img = orig_img_bgr.copy()
    color = (0, 255, 0) if label == 'no_tumor' else (0, 0, 255)
    cv2.putText(img, f"{emoji} {label.replace('_', ' ').title()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    save_path = os.path.join(full_upload_dir, filename)

    if img.shape[1] > MAX_SAVED_IMAGE_WIDTH_PX:
        scale_factor = MAX_SAVED_IMAGE_WIDTH_PX / img.shape[1]
        img = cv2.resize(img, (MAX_SAVED_IMAGE_WIDTH_PX, int(img.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)
    else:
        img = img.copy()

    cv2.imwrite(save_path, img)
    return save_path, 'green' if label == 'no_tumor' else 'red'


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/status")
async def status():
    """Returns the status of the FastAPI application and model loading."""
    return JSONResponse({"status": "ok", "model_loaded": model is not None})


@app.post("/predict")
async def predict(images: list[UploadFile] = File(...)):
    """
    Receives uploaded images, performs tumor prediction, generates Grad-CAM heatmaps,
    and returns results including paths to generated images.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"message": "Model not loaded. Please check server logs."})

    results = []
    for image in images:
        ext = os.path.splitext(image.filename)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue

        original_filename = image.filename
        gradcam_filename = f"gradcam_{os.path.splitext(original_filename)[0]}.png"

        original_image_path = os.path.join(full_upload_dir, original_filename)
        gradcam_image_path = os.path.join(full_gradcam_dir, gradcam_filename)

        try:
            data = await image.read()
            nparr = np.frombuffer(data, np.uint8)
            original_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if original_bgr is None:
                continue

            if original_bgr.shape[1] > MAX_SAVED_IMAGE_WIDTH_PX:
                scale_factor = MAX_SAVED_IMAGE_WIDTH_PX / original_bgr.shape[1]
                original_bgr_resized_for_save = cv2.resize(original_bgr,
                                                            (MAX_SAVED_IMAGE_WIDTH_PX, int(original_bgr.shape[0] * scale_factor)),
                                                            interpolation=cv2.INTER_AREA)
            else:
                original_bgr_resized_for_save = original_bgr.copy()

            cv2.imwrite(original_image_path, original_bgr_resized_for_save)

        except Exception as e:
            print(f"Error saving original image {original_filename}: {e}")
            continue

        try:
            img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            img_resized_for_model = cv2.resize(img_rgb, IMG_SIZE)
            arr = np.expand_dims(tf.keras.applications.mobilenet_v2.preprocess_input(img_resized_for_model), axis=0)

            preds = model.predict(arr)[0]
            idx = int(np.argmax(preds))
            label = label_map[idx]
            emoji = '✅' if label == 'no_tumor' else '❌'
            prob = round(float(preds[idx]) * 100, 2)

            heatmap, _ = make_gradcam_heatmap(arr, model, last_conv_layer)
            heatmap_resized = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

            if superimposed_img.shape[1] > MAX_SAVED_IMAGE_WIDTH_PX:
                scale_factor = MAX_SAVED_IMAGE_WIDTH_PX / superimposed_img.shape[1]
                superimposed_img_resized_for_save = cv2.resize(superimposed_img, (
                MAX_SAVED_IMAGE_WIDTH_PX, int(superimposed_img.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)
            else:
                superimposed_img_resized_for_save = superimposed_img.copy()

            cv2.imwrite(gradcam_image_path, superimposed_img_resized_for_save)

            results.append({
                "filename": original_filename,
                "prediction": label,
                "emoji": emoji,
                "probability": prob,
                "color": 'green' if label == 'no_tumor' else 'red',
                "gradcam": gradcam_filename
            })

        except Exception as e:
            print(f"Error processing image {original_filename} during prediction: {e}")
            continue

    return JSONResponse(results)


async def delete_temp_file(path: str):
    """Deletes a file after the response is sent."""
    try:
        os.unlink(path)
    except OSError:
        pass


@app.post("/download_pdf")
async def download_pdf(data: dict, background_tasks: BackgroundTasks):
    """
    Generates a PDF report containing original and Grad-CAM images with prediction details.
    """
    temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir())
    temp_pdf_path = temp_pdf_file.name
    temp_pdf_file.close()

    doc = SimpleDocTemplate(temp_pdf_path)
    elems = []

    elems.append(Paragraph("Brain Tumor Classification Report", styles['h1']))
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(Paragraph(
        "This report provides the classification results for the uploaded brain MRI scans, including the predicted tumor type, its confidence, Grad-CAM visualization, and general recommendations.",
        styles['ResultText']))
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(Paragraph("<hr/>", styles['Normal']))
    elems.append(Spacer(1, 0.3 * inch))

    for i, item in enumerate(data.get('results', [])):
        original_img_full_path = os.path.join(full_upload_dir, item['filename'])
        gradcam_img_full_path = os.path.join(full_gradcam_dir, item['gradcam'])

        elems.append(Paragraph(f"<b>--- Scan {i + 1}: {item['filename']} ---</b>", styles['h2']))
        elems.append(Spacer(1, 0.1 * inch))

        diagnosis_text = f"{item['emoji']} {item['prediction'].replace('_', ' ').title()}"
        confidence_text = f"Confidence: {item['probability']}%"

        if item['prediction'] == 'no_tumor':
            elems.append(Paragraph(f"<b>Classification:</b> <font color='green'>{diagnosis_text}</font>", styles['ResultText']))
            elems.append(Paragraph(f"<b>{confidence_text}</b>", styles['ResultText']))
            elems.append(Paragraph(f"Based on the analysis, <b>no tumor</b> was detected in this scan.", styles['ResultText']))
        else:
            elems.append(Paragraph(f"<b>Classification:</b> <font color='red'>{diagnosis_text}</font>", styles['ResultText']))
            elems.append(Paragraph(f"<b>{confidence_text}</b>", styles['ResultText']))
            elems.append(Paragraph(f"A <b>{item['prediction'].replace('_', ' ').title()}</b> tumor was detected in this scan.",
                                   styles['ResultText']))

        elems.append(Spacer(1, 0.1 * inch))

        predicted_label = item['prediction']
        description = tumor_descriptions.get(predicted_label, "No specific description available for this tumor type.")
        elems.append(Paragraph(f"<b>Description:</b> {description}", styles['DescriptionText']))
        elems.append(Spacer(1, 0.2 * inch))

        table_data = []
        original_img_rl = None
        gradcam_img_rl = None

        if os.path.exists(original_img_full_path):
            try:
                original_img_rl = RLImage(original_img_full_path, width=PDF_IMAGE_DISPLAY_WIDTH,
                                          height=PDF_IMAGE_DISPLAY_HEIGHT)
            except Exception as e:
                print(f"Error loading original image {original_img_full_path} for PDF: {e}")
                original_img_rl = Paragraph(f"<i>(Error loading original image)</i>", styles['SmallItalic'])
        else:
            original_img_rl = Paragraph(f"<i>(Original image not found)</i>", styles['SmallItalic'])

        if os.path.exists(gradcam_img_full_path):
            try:
                gradcam_img_rl = RLImage(gradcam_img_full_path, width=PDF_IMAGE_DISPLAY_WIDTH,
                                         height=PDF_IMAGE_DISPLAY_WIDTH)
            except Exception as e:
                print(f"Error loading Grad-CAM image {gradcam_img_full_path} for PDF: {e}")
                gradcam_img_rl = Paragraph(f"<i>(Error loading Grad-CAM image)</i>", styles['SmallItalic'])
        else:
            gradcam_img_rl = Paragraph(f"<i>(Grad-CAM image not found)</i>", styles['SmallItalic'])

        table_data.append([
            Paragraph("<b>Input MRI Scan:</b>", styles['ImageHeading']),
            Paragraph("<b>Grad-CAM Visualization:</b>", styles['ImageHeading'])
        ])
        table_data.append([original_img_rl, gradcam_img_rl])

        image_table_style = TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('LEFTPADDING', (0, 0), (0, -1), 0),
            ('RIGHTPADDING', (0, 0), (0, -1), 0),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
        ])

        image_table = Table(table_data, colWidths=[PDF_IMAGE_DISPLAY_WIDTH, PDF_IMAGE_DISPLAY_WIDTH])
        image_table.setStyle(image_table_style)
        elems.append(image_table)

        elems.append(Spacer(1, 0.1 * inch))
        elems.append(Paragraph(
            "<i>The Grad-CAM heatmap highlights regions in the image that were most influential in the model's prediction. Redder areas indicate higher importance.</i>",
            styles['SmallItalic']))
        elems.append(Spacer(1, 0.2 * inch))

        advice = treatment_advice.get(predicted_label,
                                      "No specific treatment advice available for this classification. Please consult a medical professional for guidance.")
        elems.append(Paragraph(advice, styles['TreatmentAdvice']))

        if i < len(data.get('results', [])) - 1:
            elems.append(Spacer(1, 0.3 * inch))
            elems.append(Paragraph("<hr/>", styles['Normal']))
            elems.append(Spacer(1, 0.3 * inch))
        else:
            elems.append(Spacer(1, 0.3 * inch))

    try:
        doc.build(elems)
        background_tasks.add_task(delete_temp_file, temp_pdf_path)
        response = FileResponse(temp_pdf_path, media_type='application/pdf', filename="brain_tumor_results.pdf")
        return response
    except Exception as e:
        print(f"Error building PDF: {e}")
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        return JSONResponse(status_code=500, content={"message": "Failed to generate PDF."})


@app.post("/clear-temp")
async def clear_temp():
    """Clears all uploaded and generated temporary image files."""
    for folder in [full_upload_dir, full_gradcam_dir]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
    return JSONResponse(content={"detail": "Temporary files cleared."})


def get_local_lan_ip():
    """
    Attempts to get the local machine's non-loopback IP address
    that is likely used for LAN communication.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect to the target, just sets up a connection
        # to determine the local IP that would be used for reaching it.
        # Using a public Google DNS server as a reliable target.
        s.connect(("8.8.8.8", 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'  # Fallback to loopback
    finally:
        s.close()
    return IP


def get_public_ip_address():
    """
    Attempts to get the public IP address of the machine.
    Requires an active internet connection to query an external service.
    """
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        public_ip = response.json().get('ip')
        if public_ip:
            return public_ip
    except requests.exceptions.RequestException as e:
        print(f"\nWarning: Could not fetch public IP address (Is server connected to internet?): {e}")
    return None  # Return None if unable to fetch


def open_browser(use_https: bool = False, ngrok_url: str = None):
    """
    Polls the server status and then opens the web application in a browser tab.
    Prioritizes Ngrok URL for external access if provided.
    """
    # Dynamically get the current local LAN IP for display purposes
    current_lan_ip = get_local_lan_ip()

    protocol = "https" if use_https else "http"

    # The local browser will always open 127.0.0.1
    local_browser_url = f"{protocol}://127.0.0.1:{APP_PORT}"

    # Status check uses localhost as it's always stable for local server check
    local_status_url = f"{protocol}://127.0.0.1:{APP_PORT}/status"

    max_retries = 30
    retry_delay = 1

    server_ready = False
    print("\nWaiting for server to start and respond on HTTP...")
    for i in range(max_retries):
        try:
            response = requests.get(local_status_url, timeout=retry_delay, verify=False)
            if response.status_code == 200 and response.json().get("status") == "ok":
                server_ready = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(retry_delay)

    public_ip = get_public_ip_address()  # Attempt to get public IP here

    print("\n" + "=" * 80)
    print("           🎉 BRAIN TUMOR CLASSIFICATION APP IS STARTING! 🎉")
    print("=" * 80 + "\n")

    if server_ready:
        print("\n" + "=" * 100)
        print(f"--- Application Ready (HTTP) ---")
        print(f"🔗 Local Access (from this machine): {local_browser_url}")
        # Always display LAN IP for other devices
        if current_lan_ip and current_lan_ip != '127.0.0.1':
            print(
                f"🔗 Local Network Access (from other devices on your LAN): {protocol}://{current_lan_ip}:{APP_PORT}")
        else:
            print("🔗 Local Network Access: Your machine's LAN IP could not be automatically determined.")
            print(
                "   Other devices may need to manually find it (e.g., via 'ipconfig' on Windows) to access on LAN.")

        if ngrok_url:
            print(f"🔗 External Access (via Ngrok): {ngrok_url}")
        else:
            print("\nNgrok is not configured or failed to start. External access is not available.")
        print("\n" + "=" * 100)
        print("\n" + "-" * 80)
        print("             ⚠️ IMPORTANT SECURITY NOTICE FOR Local Network Access Runners⚠️")
        print("This application is currently running on HTTP (not HTTPS).")
        print("This means the connection is NOT encrypted. Your browser will show")
        print("a 'Not secure' warning. This is expected for local development.")
        print("-" * 80 + "\n")

    else:
        print(f"--- Server Startup Issue (HTTP) ---")
        print(f"Server did not become fully ready after {max_retries} attempts.")
        print(f"This often indicates that another service is using port {APP_PORT} or there's a firewall issue.")
        print("Please ensure no other application is using this port and check your firewall settings.")
        print(f"\nAttempting to open browser anyway at: {local_browser_url}.")
        print("If the browser shows an error or cannot connect, the server is NOT running correctly.")
        print(f"\nManual local access (if server starts later): {local_browser_url}")
        if current_lan_ip and current_lan_ip != '127.0.0.1':
            print(f"Manual Local Network Access: {protocol}://{current_lan_ip}:{APP_PORT}")
        if ngrok_url:
            print(f"Manual External Access (via Ngrok): {ngrok_url}")

        print("\n" + "=" * 80)

    # Open browser with the local_browser_url
    browser_url = local_browser_url

    system = platform.system()
    chrome_paths = []

    if system == "Windows":
        chrome_paths = [
            "C:/Program Files/Google/Chrome/Application/chrome.exe",
            "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
            f"{os.environ.get('LOCALAPPDATA')}/Google/Chrome/Application/chrome.exe"
        ]
    elif system == "Darwin":  # macOS
        chrome_paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    elif system == "Linux":
        chrome_paths = [
            shutil.which("google-chrome"),
            shutil.which("chromium-browser"),
            shutil.which("chromium")
        ]

    temp_profile_dir = os.path.join(tempfile.gettempdir(), "temp_chrome_profile_" + str(os.getpid()))
    os.makedirs(temp_profile_dir, exist_ok=True)

    opened_successfully = False
    for path in chrome_paths:
        if path and os.path.exists(path):
            try:
                with open(os.devnull, 'w') as FNULL:
                    subprocess.Popen([
                        path,
                        "--new-window",
                        "--start-maximized",
                        f"--user-data-dir={temp_profile_dir}",
                        browser_url
                    ], stdout=FNULL, stderr=FNULL)
                opened_successfully = True
                break
            except Exception:
                pass

    if not opened_successfully:
        webbrowser.open_new(browser_url)


# New function to monitor LAN IP changes
def monitor_lan_ip(initial_lan_ip: str, app_port: int, ngrok_url: str): # Added ngrok_url here
    current_monitored_ip = initial_lan_ip
    protocol = "http"  # Assuming HTTP for LAN access unless HTTPS is explicitly set up

    while True:
        time.sleep(10)  # Check every 10 seconds
        new_lan_ip = get_local_lan_ip()

        if new_lan_ip != current_monitored_ip:
            print("\n" + "=" * 100)
            print(f"\n[NETWORK UPDATE] LAN IP Address has changed!")
            print(f"🔗 Local Access (from this machine): {protocol}://127.0.0.1:{app_port}")
            if new_lan_ip and new_lan_ip != '127.0.0.1':
                print(f"🔗 Local Network Access (from other devices on your LAN): {protocol}://{new_lan_ip}:{app_port}")
            else:
                print("🔗 Local Network Access: Your machine's LAN IP could not be automatically determined or is 127.0.0.1.")
                print("   Other devices may not be able to access on LAN.")

            if ngrok_url:
                print(f"🔗 External Access (via Ngrok): {ngrok_url}")
            else:
                print("🔗 External Access (via Ngrok): Not available (Ngrok not configured or failed to start).")
            print("\n" + "=" * 100)
            current_monitored_ip = new_lan_ip


if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok, conf

    ngrok_public_url = None
    initial_lan_ip = get_local_lan_ip()  # Get initial LAN IP for monitoring

    # --- Ngrok Configuration ---
    # IMPORTANT: Replace "YOUR_NGROK_AUTH_TOKEN" with your actual Ngrok authtoken.
    # Get it from: https://dashboard.ngrok.com/get-started/your-authtoken
    NGROK_AUTH_TOKEN = "2yO46OpwyDon5cHk2TANTvIyPaY_4wqhZ1cxMhZPRNhqt5Sp5" # <--- REPLACE THIS PLACEHOLDER!

    if not NGROK_AUTH_TOKEN or "YOUR_NGROK_AUTH_TOKEN" in NGROK_AUTH_TOKEN:
        print("\n" + "=" * 80)
        print("           ⚠️ NGROK AUTH TOKEN NOT SET ⚠️")
        print("To enable external access via Ngrok, please set your Ngrok authtoken.")
        print("1. Sign up/Log in to ngrok: https://dashboard.ngrok.com/signup")
        print("2. Get your authtoken: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("3. Replace 'YOUR_NGROK_AUTH_TOKEN' in the code with your actual token.")
        print("   Without it, Ngrok will not start, and only local/LAN access will be available.")
        print("=" * 80 + "\n")
        time.sleep(2)  # Give user time to read the warning
    else:
        try:
            conf.get_default().auth_token = NGROK_AUTH_TOKEN

            # 🧹 Kill old ngrok.exe (Windows only, safe to skip on other OS)
            if platform.system() == "Windows":
                try:
                    subprocess.run(["taskkill", "/f", "/im", "ngrok.exe"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                except Exception:
                    pass  # It's okay if ngrok.exe wasn't running

            # 🛑 Kill any old pyngrok tunnels
            ngrok.kill()

            # 🌍 Start new tunnel
            print("\nStarting Ngrok tunnel...")
            # Use bind_tls=True to ensure HTTPS access, which is recommended for external access.
            ngrok_tunnel = ngrok.connect(APP_PORT, bind_tls=True, domain="boa-helping-preferably.ngrok-free.app")
            ngrok_public_url = ngrok_tunnel.public_url
            print(f"✅ Ngrok tunnel established. Your app is publicly accessible at: {ngrok_public_url}")
        except Exception as e:
            print(f"\nERROR: Could not start Ngrok tunnel: {e}")
            print("Please ensure Ngrok is installed and your authtoken is correct.")
            print("Continuing without Ngrok; only local/LAN access will be available.")
            ngrok_public_url = None
    # --- End Ngrok Configuration ---

    # Start browser opening in a separate thread
    threading.Thread(target=open_browser, args=(False, ngrok_public_url)).start()

    # Start LAN IP monitoring in a separate daemon thread
    # CORRECTED LINE: Pass ngrok_public_url to the thread arguments
    ip_monitor_thread = threading.Thread(target=monitor_lan_ip, args=(initial_lan_ip, APP_PORT, ngrok_public_url), daemon=True)
    ip_monitor_thread.start()

    # Launch FastAPI app using uvicorn
    # Use host="0.0.0.0" to allow connections from Ngrok and other network devices.
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # LISTEN ON ALL AVAILABLE INTERFACES (including your LAN IP and 127.0.0.1)
        port=APP_PORT,
        reload=False,
        log_level="critical",
    )
