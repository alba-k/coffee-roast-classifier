# Coffee Roast Classifier

Classifies the roast level of coffee beans from a camera image into 7 classes: Verde, Claro, Medio,
MedioOscuro, Oscuro, Quemado and Fondo (empty paper, hands, other objects).
The model was trained with Google Teachable Machine and runs here with OpenCV + ONNX.

Course: [ISW-221] Inteligencia Artificial, Universidad Católica Boliviana, sede Santa Cruz, 2-2026.
Author: Mauri Yeison Alba Rodríguez (mauri.alba@ucb.edu.bo). Stage 1: take photos, train, verify.
Stage 2 (physical action with the ESP32) is not part of this delivery.

## Setup (virtual environment)
    powershell -ExecutionPolicy Bypass -File setup.ps1
    .\.venv\Scripts\Activate.ps1

## Reproduce the numbers in the paper
    python -m evidence                       # photo counts at every step -> results\pipeline_counts.csv
    python -m evidence.accuracy --model 2 --errors   # correct/total, 95 % Wilson interval, errors, leak check
    python tools\evaluate_model.py --split data\val
    python tools\evaluate_model.py --split data\test
    python tools\eval_by_background.py --split data\val    # old paper vs white sheet
    python tools\eval_by_layout.py --split data\val        # group / scattered / single bean

## Live camera
    python tools\classify_live.py --camera 0       # Q quits, S saves what the model sees

## Record photos (how the dataset was made)
    python main.py --camera 0 --label verde --layout grupo --start

## Re-create the ONNX model from the Teachable Machine export
Use a separate venv with `requirements-convert.txt` (Python 3.11), then:

    python tools\convert_tfjs.py model\source\tm-my-image-model.zip
    pip install onnxruntime
    python tools\verify_conversion.py model\source\tm-my-image-model.zip --n 12

## Layout
- `data/` the dataset: `train` (2,208), `val` (635) and `test` (315) photos, one folder per class
- `model/` `model.onnx` + `labels.txt` (+ `source/` original Teachable Machine export)
- `src/coffee_classifier/` classifier code; `src/coffee_dataset/` recording and curation code
- `tools/` evaluation, live classification, conversion and dataset tools
- `evidence/` scripts that count photos and compute every figure of the paper
- `results/` evaluation outputs (counts, errors, accuracy by background and layout)
- `paper/` the article in LaTeX (`main.tex`, `sections/`, `figures/`, `main.pdf`)
- `presentacion/` images used in the slides
- `firmware/` ESP32-S3 sketch for the LED ring
- `tests/` unit tests (`pip install pytest`, then `python -m pytest`)

## Lighting firmware (ESP32-S3)
`firmware/ring_serial_control/ring_serial_control.ino` sets the color and brightness of the 16-LED ring
(Arduino IDE, library "Adafruit NeoPixel"). Serial commands at 115200 baud:
`c R G B` | `l 0-255` | `s` (save) | `d` (defaults) | `p` (print). Factory values: R 230, G 235, B 100, brightness 230.
