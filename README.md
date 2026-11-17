# AI-Based Interview Bot for Self-Assessment

Final Year Project — BS Software Engineering, KFUEIT.

## Project Structure
```
interview-bot/
├── backend/
│   ├── app.py                     # Flask entry point
│   ├── routes/api.py              # REST API endpoints
│   ├── classes/
│   │   ├── video_processor.py     # FR1, FR2: Frame sampling (1 fps, 48x48) + WAV extraction
│   │   ├── emotion_detector.py    # FR3: 7-class CNN emotion classification
│   │   ├── audio_processor.py     # FR4: Google Speech-to-Text transcription
│   │   ├── semantic_analyzer.py   # FR5: NLTK preprocessing + TF-IDF Cosine Similarity
│   │   ├── feedback_generator.py  # FR6: Separate-metrics report + CSV history
│   │   └── user.py                # FR7: Video checks + Human-in-the-Loop validation
│   ├── models/                    # Trained Keras model + class_names.json
│   ├── data/                      # hr_qa.csv, history.csv, validation.csv
│   ├── uploads/                   # Temporary uploads (auto-deleted after analysis)
│   └── requirements.txt
├── frontend/
│   ├── templates/index.html       # Bootstrap 5 UI
│   └── static/                    # Custom CSS + Plain JS + Canvas chart
├── notebooks/                     # Colab training notebook
└── tests/test_pipeline.py         # Automated verification tests
```

## Prerequisites
- Python 3.12 (virtual environment in `backend\.venv`)
- `ffmpeg` installed and available on PATH
- Internet connection (for Google Speech-to-Text API and Bootstrap CDN)

## Run the Web Application
```powershell
cd path
.venv\Scripts\python.exe app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Run Automated Tests
```powershell
cd path
backend\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known Limitations (FYP Scope)
- Single candidate at a time; pre-recorded videos only (no live streaming).
- Facial emotion accuracy depends on front lighting and camera angle.
- TF-IDF measures keyword/term overlap and may score low on heavy paraphrasing (Human Assessor validation catches this).
- Google Speech-to-Text requires an active internet connection.







 ( Desktop ya D:\ drive mein).Extract 

python --version
cd path
.venv\Scripts\python.exe app.py
backend\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r backend\requirements.txt
pip install flask tensorflow opencv-python-headless numpy pandas scikit-learn nltk SpeechRecognition
pip install "opencv-python-headless==4.14.0.94"

python -c "import flask, tensorflow, cv2, numpy, pandas, sklearn, nltk, speech_recognition; print('ALL OK - OpenCV:', cv2.__version__)"

cd backend
python app.py

