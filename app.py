from flask import Flask, render_template, request, redirect, send_file, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import pdfplumber
from PIL import Image
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import os
import pytesseract
import io
import re
import uuid
import asyncio
import edge_tts
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- APP ----------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'secret123'
app.config['UPLOAD_FOLDER'] = 'uploads'

# ---------------- DB ----------------
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ---------------- MODELS ----------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    input_type = db.Column(db.String(20))
    summary = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- AI MODEL ----------------
#summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer
# ---------------- CONFIG ----------------
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- EXTRACT ----------------
def extract_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + " "
    except:
        return "Error reading PDF"
    return text

def extract_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
    except Exception as e:
        return f"Failed: {str(e)}"

    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.select("article p, main p") or soup.find_all('p')

    text = ""
    for p in paragraphs:
        text += p.get_text(" ", strip=True) + " "

    return clean_text(text)

def extract_image(file_path):
    try:
        img = Image.open(file_path)

        # convert to grayscale (improves OCR)
        img = img.convert('L')

        text = pytesseract.image_to_string(img,config='--oem 3 --psm 6')

        text = clean_text(text)

        # 🚫 remove garbage text
        if len(text.strip()) < 30:
            return "No meaningful text found in image."

        return text

    except Exception as e:
        return f"Error reading image: {str(e)}"

# ---------------- SUMMARIZE ----------------

def summarize_text(text, format_type):
     
    text = clean_text(text)
    summarizer = get_summarizer()

    # 🚫 block useless or noisy input
    if not text or len(text.split()) < 10:
        return "⚠️ Not enough readable content to summarize."

    # 🚫 remove OCR junk patterns
    if any(word in text.lower() for word in [
        "error reading image",
        "mailonline",
        "suicideprevention",
        "http"
    ]):
        return "⚠️ Extracted text looks noisy. Try a clearer image."

    # limit very large input
    if len(text) > 5000:
        text = text[:5000]

    # format tuning
    if format_type == "short":
        max_len, min_len = 60, 20
    elif format_type == "detailed":
        max_len, min_len = 180, 80
    elif format_type == "bullet":
        max_len, min_len = 120, 50
    else:
        max_len, min_len = 130, 30

    # ✅ better chunk size (prevents broken words)
    if len(text) > 800:
        chunks = []

        # split by sentences instead of raw slicing
        sentences = text.split('. ')
        temp = ""

        for sentence in sentences:
            if len(temp) + len(sentence) < 800:
                temp += sentence + ". "
            else:
                chunks.append(temp)
                temp = sentence + ". "

        if temp:
            chunks.append(temp)

        summaries = []

        for chunk in chunks:
            result = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            summaries.append(result[0]['summary_text'])

        final = " ".join(summaries)

        # second pass (clean + compress)
        result = summarizer(
            final,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return result[0]['summary_text']

    else:
        result = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return result[0]['summary_text']

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app', methods=['GET', 'POST'])
def index():
    summary = ""
    format_type = "short"

    if request.method == 'POST':
        input_type = request.form.get('type')

        if input_type == 'pdf':
            file = request.files['file']
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            text = extract_pdf(path)

        elif input_type == 'url':
            text = extract_url(request.form.get('url'))

        elif input_type == 'image':
            file = request.files['file']
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            text = extract_image(path)

        else:
            text = ""

        format_type = request.form.get('format', 'short')
        summary = summarize_text(text, format_type)

        if current_user.is_authenticated:
            db.session.add(History(
                user_id=current_user.id,
                input_type=input_type,
                summary=summary
            ))
            db.session.commit()

    history_data = []
    if current_user.is_authenticated:
        history_data = History.query.filter_by(user_id=current_user.id)\
            .order_by(History.id.desc()).all()

    return render_template('index.html',
                           summary=summary,
                           format_type=format_type,
                           history_data=history_data,
                           active_id=None)

# ---------------- LOAD HISTORY ----------------
@app.route('/load/<int:id>')
@login_required
def load_summary(id):
    item = History.query.get_or_404(id)

    if item.user_id != current_user.id:
        return "Unauthorized", 403

    history_data = History.query.filter_by(user_id=current_user.id)\
        .order_by(History.id.desc()).all()

    return render_template('index.html',
                           summary=item.summary,
                           format_type="detailed",
                           history_data=history_data,
                           active_id=id)

# ---------------- AUTH ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            return "User exists"

        user = User(username=username,
                    password=generate_password_hash(password))

        db.session.add(user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()

        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect('/app')

        return "Invalid login"

    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect('/')

#----------AUDIO--------------
@app.route('/play_audio', methods=['POST'])
def play_audio():
    try:
        summary = request.form.get('summary')
        voice = request.form.get('voice', 'female')

        if not summary or summary.strip() == "":
            return jsonify({"error": "No summary"}), 400

        voice_map = {
            "female": "en-IN-NeerjaNeural",
            "male": "en-IN-PrabhatNeural",
            "us_female": "en-US-JennyNeural",
            "us_male": "en-US-GuyNeural"
        }

        selected_voice = voice_map.get(voice, "en-IN-NeerjaNeural")

        file_path = os.path.join(app.root_path, 'static', 'output.mp3')

        async def generate():
            communicate = edge_tts.Communicate(summary, selected_voice)
            await communicate.save(file_path)

        asyncio.run(generate())

        return jsonify({"path": "/static/output.mp3"})

    except Exception as e:
        print("AUDIO ERROR:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    summary = request.form.get('summary')

    if not summary:
        return "No summary", 400

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(summary, styles["Normal"]))

    doc.build(story)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="summary.pdf",
        mimetype='application/pdf'
    )

@app.route('/delete/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    item = History.query.get_or_404(id)

    if item.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    db.session.delete(item)
    db.session.commit()

    return jsonify({"success": True})


@app.route('/delete_all', methods=['POST'])
@login_required
def delete_all_history():
    History.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()

    return jsonify({"success": True})
# ---------------- RUN ----------------
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    with app.app_context():
        db.create_all()

    app.run(debug=True)