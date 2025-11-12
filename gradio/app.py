import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import json
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.8
MAX_SEQUENCE_LENGTH = 15
REQUIRED_DETECTIONS = 2
AUDIO_PLAY_DURATION = 20

MODEL_PATH = "VGG16-va93.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_label = {str(v): k for k, v in class_indices.items()}
label_names = list(class_indices.keys())

hand_seal_images = {}
hand_seals_dir = Path("hand-seals")

for label in label_names:
    img_path = hand_seals_dir / f"{label}.jpg"
    if img_path.exists():
        hand_seal_images[label] = str(img_path)
    else:
        print(f"Warning: {img_path} not found")

def preprocess_frame(frame):
    """
    Preprocess frame to match training:
    - Resize to 224x224
    - Ensure RGB format
    - Normalize by dividing by 255.0
    - Add batch dimension
    """
    
    resized = cv2.resize(frame, (224, 224))
    

    if len(resized.shape) == 3 and resized.shape[2] == 3:
        pass
    else:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    normalized = resized.astype(np.float32) / 255.0
    batch_input = np.expand_dims(normalized, axis=0)
    return batch_input

def predict_hand_sign(frame):
    """
    Run model inference and return top prediction with confidence
    """
    preprocessed = preprocess_frame(frame)
    predictions = model.predict(preprocessed, verbose=0)
    probabilities = predictions[0]
    top_index = np.argmax(probabilities)
    top_confidence = float(probabilities[top_index])
    top_label = label_names[top_index]
    return top_label, top_confidence, probabilities

def check_chidori_sequence(history):
    """
    Check if the prediction history contains the Chidori sequence: Ox → Hare → Monkey
    Each seal must appear at least REQUIRED_DETECTIONS times in the sequence.
    """
    if len(history) < REQUIRED_DETECTIONS * 3:
        return False
    
    ox_count = 0
    hare_count = 0
    monkey_count = 0
    state = "looking_for_ox"
    
    for seal in history:
        if state == "looking_for_ox" and seal == "ox":
            ox_count += 1
            if ox_count >= REQUIRED_DETECTIONS:
                state = "looking_for_hare"
        elif state == "looking_for_hare" and seal == "hare":
            hare_count += 1
            if hare_count >= REQUIRED_DETECTIONS:
                state = "looking_for_monkey"
        elif state == "looking_for_monkey" and seal == "monkey":
            monkey_count += 1
            if monkey_count >= REQUIRED_DETECTIONS:
                return True
    
    return False

def process_frame(frame, sequence_history, audio_counter):
    if frame is None:
        return "No frame detected", None, sequence_history, audio_counter
    
    top_label, top_confidence, all_probs = predict_hand_sign(frame)
    
    if audio_counter == 0 and top_confidence >= CONFIDENCE_THRESHOLD:
        sequence_history.append(top_label)
        if len(sequence_history) > MAX_SEQUENCE_LENGTH:
            sequence_history.pop(0)

    confidence_percent = int(top_confidence * 100)
    prediction_text = f"{top_label.capitalize()}: {confidence_percent}%"
    
    top3_indices = np.argsort(all_probs)[-3:][::-1]
    top3_text = "\nTop 3 Current Predictions:\n"
    for idx in top3_indices:
        label = label_names[idx]
        conf = int(all_probs[idx] * 100)
        top3_text += f" - {label.capitalize()}: {conf}%\n"
    
    audio_file = None
    jutsu_detected = ""
    
    if audio_counter == 0 and check_chidori_sequence(sequence_history):
        audio_counter = AUDIO_PLAY_DURATION
        sequence_history.clear()
    
    if audio_counter > 0:
        audio_file = "chidori SFX.mp3"
        jutsu_detected = "\n\n⚡CHIDORI DETECTED⚡\nSequence: Ox → Hare → Monkey"
        audio_counter -= 1
    
    if len(sequence_history) > 0:
        recent_seals = " → ".join([s.capitalize() for s in sequence_history[-6:]])
        sequence_text = f"\n\nRecent Sequence:\n{recent_seals}"
    else:
        sequence_text = "\n\nRecent Sequence: (none)"
    
    full_prediction = prediction_text + "\n" + top3_text + sequence_text + jutsu_detected
    return full_prediction, audio_file, sequence_history, audio_counter

custom_css = """
    .center-title {
        text-align: center !important;
    }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
                <div style="text-align: center; margin: 0 auto; max-width: 800px;">
                    <h1> Naruto Hand Seal Demo </h1>
                    <h2>Instructions:</h2>
                    <div style="text-align: left; display: inline-block;">
                        <h4>1. Click on the Webcam Feed ('Click to Access Webcam') & connect with your desired webcam</h4>
                        <h4>2. Make sure your hands are the ONLY things in the webcam frame (for better UX)</h4>
                        <h4>3. Click on 'Record' and start signing different Seals from the Naruto anime</h4>
                    </div>
                </div>
                """)
    
    sequence_state = gr.State([])
    audio_counter_state = gr.State(0)
    
    with gr.Row():

        with gr.Column(min_width=80):
            for seal in label_names[:6]:
                if seal in hand_seal_images:
                    gr.Image(
                        value=hand_seal_images[seal],
                        width=100,
                        height=100,
                        interactive=False,
                        show_label=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                        container=False
                    )
        
        with gr.Column(scale=6):
            webcam_input = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Webcam Feed",
                streaming=True,
                height=680,
        )

        with gr.Column(min_width=80):
            for seal in label_names[6:]:
                if seal in hand_seal_images:
                    gr.Image(
                        value=hand_seal_images[seal],
                        width=100,
                        height=100,
                        interactive=False,
                        show_label=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                        container=False
                    )
        with gr.Column(scale=6):
            prediction_display = gr.Textbox(
                label="Model Prediction",
                value="Waiting for input...",
                interactive=False,
                lines=8
            )

            audio_output = gr.Audio(
                label="Jutsu SFX",
                autoplay=True,
                visible=True
            )

    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input, sequence_state, audio_counter_state],
        outputs=[prediction_display, audio_output, sequence_state, audio_counter_state],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch()
