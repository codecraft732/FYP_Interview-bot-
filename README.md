# AI-Based Interview Bot for Self-Assessment

Final Year Project — BSc Software Engineering, KFUEIT.

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
cd C:\Users\Microsoft\.gemini\antigravity\scratch\interview-bot\backend
.venv\Scripts\python.exe app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Run Automated Tests
```powershell
cd C:\Users\Microsoft\.gemini\antigravity\scratch\interview-bot
backend\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known Limitations (FYP Scope)
- Single candidate at a time; pre-recorded videos only (no live streaming).
- Facial emotion accuracy depends on front lighting and camera angle.
- TF-IDF measures keyword/term overlap and may score low on heavy paraphrasing (Human Assessor validation catches this).
- Google Speech-to-Text requires an active internet connection.


















📋 Project Run Karne Ka Mukammal Tareeqa (Step-by-Step Guide)
Step 1: ZIP Extract Karein aur Project Folder Mein Jayein
Di gayi ZIP file (interview-bot.zip) ko apne computer par kisi bhi jagah extract karein (maslan Desktop ya D:\ drive mein).

Extract hone ke baad interview-bot folder ko open karein (jahan backend, frontend, notebooks, aur tests folders nazar aa rahe hon).

Folder ke upar Address Bar par click karein, wahan powershell likh kar Enter press karein (is se PowerShell seedha usi project folder ke path par open ho jayega).

(Ya PowerShell khol kar cd ke saath folder ka path paste karein, maslan:)

PowerShell
cd "C:\Users\AapkaNaam\Desktop\interview-bot"
Step 2: Python Version Check Karein
Check karein ke system mein Python 3.12 (ya 3.10–3.12) install hai ya nahi:

PowerShell
python --version
(Note: Agar Python install nahi hai toh python.org se Python 3.12 install karein aur installation ke waqt "Add Python to PATH" par tick zaroor lagayein.)

Step 3: Virtual Environment (.venv) Banayein aur Activate Karein
Project ke root folder (interview-bot) mein rehte hue yeh commands ek-ek karke chalayein:

Virtual environment create karein:

PowerShell

Virtual environment activate karein:

PowerShell
backend\.venv\Scripts\Activate.ps1
(Agar PowerShell mein red color ka "Execution Policy" error aaye, toh pehle yeh command chala kar phir upar wali command dobara chalayein:)

PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Step 4: Saare Required Packages Install Karein
pip ko update karein aur saare zaroori packages install karein:

Pip upgrade karein:

PowerShell
python -m pip install --upgrade pip
Saare packages ek saath install karein:(Kyunki aapke paas pehle se backend\requirements.txt maujood hai, isliye woh seedha yeh chala sakte hain:)

PowerShell
pip install -r backend\requirements.txt
YA agar manually saare packages install karne hon toh yeh commands chalayein:

PowerShell
pip install flask tensorflow opencv-python-headless numpy pandas scikit-learn nltk SpeechRecognition
pip install "opencv-python-headless==4.14.0.94"
(Note: opencv-python-headless ka 4.14.0.94 version isliye zaroori hai taake cv2.CascadeClassifier (face detection) bina kisi error ke kaam kare).

Step 5: Check Karein Ke Sab Libraries Sahi Install Hui Hain
Tasalli karne ke liye yeh test command chalayein:

PowerShell
python -c "import flask, tensorflow, cv2, numpy, pandas, sklearn, nltk, speech_recognition; print('ALL OK - OpenCV:', cv2.__version__)"
Agar output mein ALL OK - OpenCV: 4.14.0 likha aa jaye toh setup mukammal hai.

Step 6: Backend Server Run Karein
Ab backend folder ke andar jayein aur app.py run karein:

Backend folder mein jayein:

PowerShell
cd backend
App start karein:

PowerShell
python app.py

