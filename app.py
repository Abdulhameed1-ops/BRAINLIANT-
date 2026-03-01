"""
Brainliant — Production-ready Flask app
Auth: Flask-Login + Flask-Bcrypt + SQLAlchemy (PostgreSQL / SQLite)
Persistent sessions, bcrypt hashing, secure cookies, ToS/Privacy routes.
"""
import os, json, re, io, zlib, uuid, datetime
from functools import wraps

from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, send_from_directory)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin,
                         login_user, logout_user, login_required,
                         current_user)
from flask_bcrypt import Bcrypt
import requests as http_req

# ─── APP FACTORY ──────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── detect production (Render sets the RENDER env var) ──
IS_PRODUCTION = bool(os.environ.get('RENDER') or os.environ.get('PRODUCTION'))

# ── Secret key — MUST be set in Render environment variables ──
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    if IS_PRODUCTION:
        raise RuntimeError('SECRET_KEY environment variable is required in production!')
    app.secret_key = 'brainliant-local-dev-only-change-me'

# ── Database URL — Render provides DATABASE_URL for PostgreSQL ──
raw_db_url = os.environ.get('DATABASE_URL', 'sqlite:///brainliant.db')
# Render (and older Heroku) uses postgres:// — SQLAlchemy requires postgresql://
if raw_db_url.startswith('postgres://'):
    raw_db_url = raw_db_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = raw_db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,       # reconnect after idle timeout
    'pool_recycle': 300,         # recycle connections every 5 min
}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# ── Session / Cookie security ──
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=30)
app.config['SESSION_COOKIE_HTTPONLY']    = True
app.config['SESSION_COOKIE_SAMESITE']   = 'Lax'
# Secure cookies only work on HTTPS — enable in production
app.config['SESSION_COOKIE_SECURE']     = IS_PRODUCTION
app.config['SESSION_COOKIE_NAME']       = 'brainliant_session'
app.config['REMEMBER_COOKIE_DURATION']  = datetime.timedelta(days=30)
app.config['REMEMBER_COOKIE_SECURE']    = IS_PRODUCTION
app.config['REMEMBER_COOKIE_HTTPONLY']  = True
app.config['REMEMBER_COOKIE_SAMESITE']  = 'Lax'

# ── Extensions ──
db      = SQLAlchemy(app)
bcrypt  = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'   # redirect unauthenticated requests to SPA

# ── App constants ──
COHERE_API_KEY   = os.environ.get('COHERE_API_KEY', '')
OCR_API_KEY      = os.environ.get('OCR_API_KEY', '')
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
RANKS            = ['Beginner','Explorer','Scholar','Strategist','Master','Grandmaster','Legend']
RANK_XP        = [0, 500, 1500, 3500, 7000, 12000, 20000]

# ─── MODELS ───────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id              = db.Column(db.String(36),  primary_key=True, default=lambda: str(uuid.uuid4()))
    name            = db.Column(db.String(120),  nullable=False)
    username        = db.Column(db.String(50),   unique=True, nullable=False)
    email           = db.Column(db.String(200),  unique=True, nullable=False)
    password_hash   = db.Column(db.String(255),  nullable=True)    # null for Google-only accounts
    google_id       = db.Column(db.String(128),  unique=True, nullable=True)   # Google sub claim
    xp              = db.Column(db.Integer,      default=0)
    level           = db.Column(db.Integer,      default=1)
    rank            = db.Column(db.String(50),   default='Beginner')
    streak          = db.Column(db.Integer,      default=1)
    last_active     = db.Column(db.Date,         default=datetime.date.today)
    courses_done    = db.Column(db.Integer,      default=0)
    quizzes_done    = db.Column(db.Integer,      default=0)
    total_time_min  = db.Column(db.Integer,      default=0)
    created_at      = db.Column(db.DateTime,     default=datetime.datetime.utcnow)

    # Flask-Login requires get_id() — already provided by UserMixin (returns self.id)

    def set_password(self, password: str):
        """Hash password with bcrypt and store it."""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password: str) -> bool:
        """
        Verify password. Handles:
        1. bcrypt hash (standard)
        2. SHA-256 hex (legacy — migrates to bcrypt on success)
        3. Google-only accounts (no password_hash) — always False
        """
        if not self.password_hash:
            return False   # Google-only account, use Google sign-in
        if self.password_hash.startswith('$2'):
            return bcrypt.check_password_hash(self.password_hash, password)
        # Legacy SHA-256 migration
        import hashlib
        sha256_hash = hashlib.sha256(password.encode()).hexdigest()
        if sha256_hash == self.password_hash:
            self.set_password(password)
            db.session.commit()
            return True
        return False

    def rank_index(self) -> int:
        return RANKS.index(self.rank) if self.rank in RANKS else 0

    def update_rank(self):
        for i in range(len(RANK_XP)-1, -1, -1):
            if self.xp >= RANK_XP[i]:
                self.rank  = RANKS[i]
                self.level = i + 1
                break

    def update_streak(self):
        today = datetime.date.today()
        if self.last_active:
            diff = (today - self.last_active).days
            if diff == 1:   self.streak = (self.streak or 0) + 1
            elif diff > 1:  self.streak = 1
        self.last_active = today

    @property
    def rank_progress(self) -> int:
        ri = self.rank_index()
        lo = RANK_XP[ri]
        hi = RANK_XP[ri + 1] if ri + 1 < len(RANK_XP) else lo + 5000
        return min(100, round((self.xp - lo) / max(hi - lo, 1) * 100))

    @property
    def global_position(self) -> int:
        return User.query.filter(User.xp > self.xp).count() + 1

    @property
    def percentile(self) -> int:
        total = User.query.count()
        return max(0, round((1 - self.global_position / max(total, 1)) * 100))

    def to_dict(self) -> dict:
        ri  = self.rank_index()
        lo  = RANK_XP[ri]
        hi  = RANK_XP[ri + 1] if ri + 1 < len(RANK_XP) else lo + 5000
        prg = min(100, round((self.xp - lo) / max(hi - lo, 1) * 100))
        # Single aggregated query instead of two separate ones
        try:
            above = User.query.filter(User.xp > self.xp).count()
            total = User.query.count()
            pos   = above + 1
            pct   = max(0, round((1 - pos / max(total, 1)) * 100))
        except Exception:
            pos, pct = 1, 100
        return {
            'id':              self.id,
            'name':            self.name,
            'username':        self.username,
            'email':           self.email,
            'xp':              self.xp,
            'level':           self.level,
            'rank':            self.rank,
            'streak':          self.streak,
            'courses':         self.courses_done,
            'quizzes_done':    self.quizzes_done,
            'total_time_min':  self.total_time_min,
            'global_position': pos,
            'percentile':      pct,
            'rank_progress':   prg,
            'next_rank':       RANKS[min(ri + 1, len(RANKS) - 1)],
            'next_rank_xp':    RANK_XP[min(ri + 1, len(RANK_XP) - 1)],
        }


class TopicMastery(db.Model):
    __tablename__ = 'topic_mastery'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    topic           = db.Column(db.String(200), nullable=False)
    correct         = db.Column(db.Integer, default=0)
    total           = db.Column(db.Integer, default=0)
    last_seen       = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    next_review     = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    interval_days   = db.Column(db.Integer, default=1)
    __table_args__  = (db.UniqueConstraint('user_id', 'topic'),)

    def mastery_pct(self) -> int:
        return round(self.correct / max(self.total, 1) * 100)

    def schedule_next(self, was_correct: bool):
        self.interval_days = min(self.interval_days * 2, 30) if was_correct else 1
        self.next_review   = datetime.datetime.utcnow() + datetime.timedelta(days=self.interval_days)
        self.last_seen     = datetime.datetime.utcnow()

    def to_dict(self) -> dict:
        return {'topic': self.topic, 'correct': self.correct, 'total': self.total,
                'mastery_pct': self.mastery_pct()}


class StudySession(db.Model):
    __tablename__ = 'study_sessions'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    date            = db.Column(db.Date, default=datetime.date.today)
    duration        = db.Column(db.Integer, default=0)
    xp_earned       = db.Column(db.Integer, default=0)
    topics          = db.Column(db.Text, default='[]')
    # Daily activity counters — tracked to gate the daily streak
    files_uploaded  = db.Column(db.Integer, default=0)   # upload page file extractions
    hw_images       = db.Column(db.Integer, default=0)   # homework explain + check calls
    quizzes_done    = db.Column(db.Integer, default=0)   # quiz completions
    __table_args__  = (db.UniqueConstraint('user_id', 'date'),)

    # A streak-day is "secured" when the user hits 5 activities total
    # (any combination of files uploaded, homework images, quizzes)
    DAILY_GOAL = 5

    def activities_today(self) -> int:
        return (self.files_uploaded or 0) + (self.hw_images or 0) + (self.quizzes_done or 0)

    def streak_secured(self) -> bool:
        return self.activities_today() >= self.DAILY_GOAL


class SavedQuiz(db.Model):
    __tablename__ = 'saved_quizzes'
    id          = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title       = db.Column(db.String(200), nullable=False)
    questions   = db.Column(db.Text, nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class SystemStats(db.Model):
    __tablename__ = 'system_stats'
    id              = db.Column(db.Integer, primary_key=True)
    total_users     = db.Column(db.Integer, default=0)
    total_quizzes   = db.Column(db.Integer, default=0)
    total_files     = db.Column(db.Integer, default=0)


# ─── FLASK-LOGIN ──────────────────────────────────────────────────────────────

@login_manager.user_loader
def load_user(user_id: str):
    """
    Called on every request to restore the user from the session cookie.
    Uses db.session.get() — the SQLAlchemy 2.0 safe way (User.query.get is deprecated).
    """
    try:
        return db.session.get(User, str(user_id))
    except Exception:
        return None

@login_manager.unauthorized_handler
def unauthorized():
    """API endpoints return 401; page routes redirect to SPA."""
    if request.path.startswith('/auth/') or request.path.startswith('/ai/') \
            or request.path.startswith('/user/') or request.path.startswith('/quiz/') \
            or request.path.startswith('/upload/') or request.path in ('/leaderboard', '/stats'):
        return jsonify({'error': 'Not authenticated — please log in'}), 401
    return redirect(url_for('index'))

# ─── SESSION PERSISTENCE ─────────────────────────────────────────────────────

@app.before_request
def make_session_permanent():
    """
    Mark every session as permanent so Flask honours PERMANENT_SESSION_LIFETIME.
    Without this, sessions are browser-session only (expire on tab close).
    Flask-Login's remember=True also sets a separate long-lived cookie as a
    second layer of protection.
    """
    from flask import session as _session
    _session.permanent = True

# ─── DB INIT & MIGRATION ──────────────────────────────────────────────────────

with app.app_context():
    db.create_all()

    # Safe column migrations — add any new columns that may not exist yet
    # in databases created before these columns were introduced.
    # This runs every startup and is idempotent (safe to run repeatedly).
    try:
        with db.engine.connect() as conn:
            # Detect dialect
            dialect = db.engine.dialect.name  # 'postgresql' or 'sqlite'

            def column_exists(table, column):
                if dialect == 'postgresql':
                    result = conn.execute(
                        db.text("SELECT 1 FROM information_schema.columns "
                                "WHERE table_name=:t AND column_name=:c"),
                        {'t': table, 'c': column}
                    )
                else:  # sqlite
                    result = conn.execute(db.text(f"PRAGMA table_info({table})"))
                    return any(row[1] == column for row in result)
                return result.fetchone() is not None

            migrations = [
                # (table, column, DDL to add it)
                ('users', 'google_id',
                 'ALTER TABLE users ADD COLUMN google_id VARCHAR(128)'),
                ('study_sessions', 'files_uploaded',
                 'ALTER TABLE study_sessions ADD COLUMN files_uploaded INTEGER DEFAULT 0'),
                ('study_sessions', 'hw_images',
                 'ALTER TABLE study_sessions ADD COLUMN hw_images INTEGER DEFAULT 0'),
                ('study_sessions', 'quizzes_done',
                 'ALTER TABLE study_sessions ADD COLUMN quizzes_done INTEGER DEFAULT 0'),
            ]

            for table, column, ddl in migrations:
                if not column_exists(table, column):
                    conn.execute(db.text(ddl))
                    conn.commit()

    except Exception as migration_err:
        # Never let a migration failure crash the app
        print(f'[migration] warning: {migration_err}')

    if not SystemStats.query.first():
        db.session.add(SystemStats())
        db.session.commit()

# ─── COHERE ───────────────────────────────────────────────────────────────────

def cohere(prompt: str, max_tok: int = 4096) -> str:
    if not COHERE_API_KEY:
        raise ValueError(
            'COHERE_API_KEY is not set. '
            'Add it to Render → Environment Variables.'
        )
    try:
        r = http_req.post(
            'https://api.cohere.ai/v1/chat',
            headers={'Authorization': f'Bearer {COHERE_API_KEY}',
                     'Content-Type': 'application/json'},
            json={'model': 'command-a-03-2025', 'message': prompt,
                  'max_tokens': max_tok, 'temperature': 0.7},
            timeout=90
        )
    except http_req.exceptions.ConnectionError:
        raise ValueError('Cannot reach Cohere AI — check your internet connection.')
    except http_req.exceptions.Timeout:
        raise ValueError('Cohere AI timed out — please try again.')

    if r.status_code == 401:  raise ValueError('Invalid COHERE_API_KEY.')
    if r.status_code == 429:  raise ValueError('Cohere rate limit hit — wait a moment.')
    if not r.ok:              raise ValueError(f'Cohere error {r.status_code}: {r.text[:200]}')

    text = r.json().get('text', '')
    if not text:
        raise ValueError('Cohere returned an empty response — please try again.')
    return text

# ─── OCR ──────────────────────────────────────────────────────────────────────

def ocr_image(file_obj, fname: str = 'image.png'):
    """Returns (text, error_or_None). Uses multipart upload (more reliable than base64)."""
    if not OCR_API_KEY:
        return '', 'OCR_API_KEY not set in environment variables.'
    try:
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        raw = file_obj.read() if hasattr(file_obj, 'read') else file_obj
        # Normalise with PIL for better accuracy
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            buf = io.BytesIO(); img.save(buf, format='PNG')
            raw = buf.getvalue(); fname = 'image.png'
        except Exception:
            pass
        r = http_req.post('https://api.ocr.space/parse/image',
                          data={'apikey': OCR_API_KEY, 'language': 'eng',
                                'isOverlayRequired': False, 'detectOrientation': True,
                                'scale': True, 'OCREngine': 2},
                          files={'file': (fname, raw, 'image/png')}, timeout=30)
        if r.status_code != 200:
            return '', f'OCR service returned HTTP {r.status_code}'
        data = r.json()
        if data.get('IsErroredOnProcessing'):
            msgs = data.get('ErrorMessage', ['OCR error'])
            return '', msgs[0] if isinstance(msgs, list) else str(msgs)
        parsed = data.get('ParsedResults', [])
        if not parsed:
            return '', 'No text detected in image.'
        text = parsed[0].get('ParsedText', '').strip()
        return (text, None) if text else ('', 'Image contained no readable text.')
    except http_req.exceptions.Timeout:
        return '', 'OCR timed out — please try again.'
    except Exception as e:
        return '', f'OCR error: {e}'

# ─── FILE EXTRACTORS ──────────────────────────────────────────────────────────

def ext_pdf(data: bytes) -> str:
    """Extract text from PDF using pypdf — handles text-based PDFs properly."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t and t.strip():
                    pages.append(t.strip())
            except Exception:
                pass
        text = '\n\n'.join(pages)
        return re.sub(r'[ \t]+', ' ', text).strip()
    except Exception:
        return ''

def ext_docx(data: bytes) -> str:
    try:
        import docx
        return '\n'.join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs if p.text.strip())
    except Exception:
        pass
    try:
        import zipfile
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open('word/document.xml') as f:
                ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                return ' '.join(e.text for e in ET.parse(f).getroot().iter(ns+'t') if e.text)
    except Exception:
        return ''

def ext_xlsx(data: bytes) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data)); parts = []
        for sh in wb.worksheets:
            parts.append(f'[Sheet:{sh.title}]')
            for row in sh.iter_rows(values_only=True):
                t = ' | '.join(str(c) for c in row if c is not None)
                if t.strip(): parts.append(t)
        return '\n'.join(parts)
    except Exception:
        return ''

def ext_audio(data: bytes, filename: str = 'audio.wav') -> str:
    """Transcribe audio using SpeechRecognition (Google free API). Converts via pydub if needed."""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        fn_lower = filename.lower()
        audio_data = data
        if not fn_lower.endswith('.wav'):
            try:
                from pydub import AudioSegment
                fmt_map = {'.mp3': 'mp3', '.m4a': 'mp4', '.ogg': 'ogg',
                           '.webm': 'webm', '.flac': 'flac', '.aac': 'aac'}
                ext = '.' + fn_lower.rsplit('.', 1)[-1] if '.' in fn_lower else '.mp3'
                fmt = fmt_map.get(ext, 'mp3')
                seg = AudioSegment.from_file(io.BytesIO(data), format=fmt)
                wav_buf = io.BytesIO()
                seg.export(wav_buf, format='wav')
                wav_buf.seek(0)
                audio_data = wav_buf.read()
            except Exception:
                return ''
        with sr.AudioFile(io.BytesIO(audio_data)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception:
        return ''


def proc_file(fs):
    """Route uploaded FileStorage to the right extractor. Returns (text, error_or_None)."""
    fn = (fs.filename or '').lower()
    data = fs.read()
    if fn.endswith('.pdf'):
        t = ext_pdf(data)
        if len(t.strip()) >= 30:
            return (t, None)
        return (None, 'PDF has no selectable text — it may be scanned. Try a text-based PDF or add the topic in the Title field.')
    if fn.endswith(('.docx', '.doc')):
        t = ext_docx(data); return (t, None) if t.strip() else (None, 'Could not read DOCX file.')
    if fn.endswith(('.xlsx', '.xls')):
        t = ext_xlsx(data); return (t, None) if t.strip() else (None, 'Could not read Excel file.')
    if fn.endswith(('.png','.jpg','.jpeg','.gif','.bmp','.tiff','.webp')):
        t, err = ocr_image(io.BytesIO(data), fs.filename)
        return (t, None) if t else (None, err or 'OCR failed.')
    if fn.endswith(('.txt', '.md', '.csv')):
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try: return data.decode(enc), None
            except: pass
        return data.decode('utf-8', errors='ignore'), None
    if fn.endswith(('.mp3', '.wav', '.ogg', '.m4a', '.webm', '.flac', '.aac')):
        t = ext_audio(data, fs.filename)
        return (t, None) if t.strip() else (None, 'Could not transcribe audio — ensure the file has clear speech.')
    return None, f'Unsupported file type: {fn.rsplit(".", 1)[-1] if "." in fn else "unknown"}'

# ─── ADAPTIVE LEARNING HELPERS ────────────────────────────────────────────────

def get_weak_topics(user_id: str, n: int = 5) -> list:
    rows = TopicMastery.query.filter_by(user_id=user_id).filter(TopicMastery.total > 0).all()
    rows.sort(key=lambda t: t.correct / max(t.total, 1))
    return [r.topic for r in rows[:n]]

def get_due_topics(user_id: str) -> list:
    return TopicMastery.query.filter_by(user_id=user_id)\
        .filter(TopicMastery.next_review <= datetime.datetime.utcnow()).all()

def upsert_topic(user_id: str, topic: str, correct: int, total: int):
    tm = TopicMastery.query.filter_by(user_id=user_id, topic=topic).first()
    if not tm:
        tm = TopicMastery(user_id=user_id, topic=topic)
        db.session.add(tm)
    tm.correct += correct
    tm.total   += total
    tm.schedule_next(correct > 0)
    db.session.commit()

def fallback_qs(topic: str, n: int) -> list:
    base = [
        {'q': f'Core concept of "{topic}"?',
         'opts': ['A) Main framework','B) Minor detail','C) Unrelated idea','D) Deprecated concept'],
         'ans': 'A', 'exp': 'Understanding the core framework is essential.', 'topic': topic},
        {'q': f'Best way to study "{topic}"?',
         'opts': ['A) Re-reading','B) Active recall','C) Memorise only','D) Skip hard parts'],
         'ans': 'B', 'exp': 'Active recall builds far stronger memory than passive review.', 'topic': topic},
        {'q': f'How to handle difficult "{topic}" sections?',
         'opts': ['A) Skip entirely','B) Break into chunks','C) Read faster','D) Summaries only'],
         'ans': 'B', 'exp': 'Chunking reduces cognitive load and makes complex material manageable.', 'topic': topic},
    ]
    return [dict(base[i % len(base)], q=f'Q{i+1}: {base[i%len(base)]["q"]}') for i in range(n)]

# ─── DAILY ACTIVITY HELPER ───────────────────────────────────────────────────

def get_or_create_today_session(user_id: str) -> StudySession:
    """Get today's StudySession, creating it if it doesn't exist."""
    today = datetime.date.today()
    sess = StudySession.query.filter_by(user_id=user_id, date=today).first()
    if not sess:
        sess = StudySession(user_id=user_id, date=today)
        db.session.add(sess)
        db.session.flush()   # get the id without committing
    return sess

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/static/<path:fn>')
def statics(fn):
    return send_from_directory(os.path.join(app.root_path, 'static'), fn)

@app.route('/ping')
def ping():
    return jsonify({'ok': True, 'cohere': bool(COHERE_API_KEY), 'ocr': bool(OCR_API_KEY),
                    'production': IS_PRODUCTION})

@app.route('/user/daily-status')
@login_required
def daily_status():
    """Returns today's activity counts and whether the streak is secured."""
    today = datetime.date.today()
    sess = StudySession.query.filter_by(user_id=current_user.id, date=today).first()
    files    = sess.files_uploaded if sess else 0
    hw       = sess.hw_images      if sess else 0
    quizzes  = sess.quizzes_done   if sess else 0
    total    = files + hw + quizzes
    secured  = total >= StudySession.DAILY_GOAL
    return jsonify({
        'files_uploaded': files,
        'hw_images':      hw,
        'quizzes_done':   quizzes,
        'total_today':    total,
        'daily_goal':     StudySession.DAILY_GOAL,
        'streak_secured': secured,
        'streak':         current_user.streak,
    })

@app.route('/stats')
def stats():
    # Always count real users directly from the users table — never fake numbers
    real_users   = User.query.count()
    # Quizzes and files come from SystemStats (incremented on each action)
    s = SystemStats.query.first()
    total_quizzes = s.total_quizzes if s else 0
    total_files   = s.total_files   if s else 0
    return jsonify({
        'users':   real_users,
        'quizzes': total_quizzes,
        'files':   total_files,
    })

# ── Legal pages ───────────────────────────────────────────────────────────────

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route('/auth/google', methods=['POST'])
def google_auth():
    """
    Verify a Google ID token sent from the frontend.
    Creates a new account or logs into an existing one.
    No GOOGLE_CLIENT_ID needed for token verification — we use Google's tokeninfo endpoint.
    If GOOGLE_CLIENT_ID is set, we also verify the audience claim for extra security.
    """
    d     = request.get_json() or {}
    token = (d.get('credential') or '').strip()
    if not token:
        return jsonify({'error': 'No Google credential received.'}), 400

    # Verify the token with Google's public endpoint
    try:
        r = http_req.get(
            'https://oauth2.googleapis.com/tokeninfo',
            params={'id_token': token},
            timeout=10
        )
        if not r.ok:
            return jsonify({'error': 'Google sign-in failed — invalid token.'}), 401
        info = r.json()
    except Exception:
        return jsonify({'error': 'Could not reach Google to verify sign-in.'}), 503

    # Validate audience if GOOGLE_CLIENT_ID is configured
    if GOOGLE_CLIENT_ID and info.get('aud') != GOOGLE_CLIENT_ID:
        return jsonify({'error': 'Google sign-in failed — token audience mismatch.'}), 401

    google_id = info.get('sub', '')
    email     = (info.get('email') or '').strip().lower()
    name      = (info.get('name') or email.split('@')[0] or 'Learner').strip()

    if not google_id or not email:
        return jsonify({'error': 'Google did not return required profile information.'}), 400

    # Find existing user by google_id or email
    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User.query.filter_by(email=email).first()

    if user:
        # Existing user — link Google ID if not already linked
        if not user.google_id:
            user.google_id = google_id
        user.update_streak()
        db.session.commit()
    else:
        # New user — create account
        base_username = re.sub(r'[^a-z0-9]', '', email.split('@')[0].lower()) or 'user'
        username = base_username
        counter  = 1
        while User.query.filter_by(username=username).first():
            username = f'{base_username}{counter}'; counter += 1

        user = User(name=name, username=username, email=email, google_id=google_id)
        # No password for Google accounts — password_hash stays None
        db.session.add(user)
        stats = SystemStats.query.first()
        if stats: stats.total_users += 1
        db.session.commit()

    from flask import session as _session
    _session.permanent = True
    login_user(user, remember=True)

    is_new = not bool(info.get('sub') and User.query.filter_by(google_id=google_id).count() > 1)
    return jsonify({'user': user.to_dict(), 'message': 'Signed in with Google!'})


@app.route('/app/download')
def app_download():
    """Serve the Android APK for download."""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'Brainliant.apk',
        as_attachment=True,
        download_name='Brainliant.apk'
    )


@app.route('/auth/check')
def auth_check():
    if current_user.is_authenticated:
        current_user.update_streak()
        db.session.commit()
        return jsonify({'authenticated': True, 'user': current_user.to_dict()})
    return jsonify({'authenticated': False})

@app.route('/auth/register', methods=['POST'])
def register():
    d        = request.get_json() or {}
    name     = (d.get('name') or '').strip()
    username = (d.get('username') or '').strip().lower()
    email    = (d.get('email') or '').strip().lower()
    password = (d.get('password') or '')

    if not all([name, username, email, password]):
        return jsonify({'error': 'All fields are required.'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters.'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters.'}), 400
    if not re.match(r'^[a-z0-9_]+$', username):
        return jsonify({'error': 'Username may only contain letters, numbers, and underscores.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'That email address is already registered.'}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'That username is already taken.'}), 409

    user = User(name=name, username=username, email=email)
    user.set_password(password)       # bcrypt hash
    db.session.add(user)

    stats = SystemStats.query.first()
    if stats: stats.total_users += 1
    db.session.commit()

    # Log the user in immediately after registration with persistent remember-me cookie
    from flask import session as _session
    _session.permanent = True
    login_user(user, remember=True)

    return jsonify({'user': user.to_dict(), 'message': 'Account created successfully!'}), 201

@app.route('/auth/login', methods=['POST'])
def login():
    d        = request.get_json() or {}
    email    = (d.get('email') or '').strip().lower()
    password = (d.get('password') or '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required.'}), 400

    user = User.query.filter_by(email=email).first()

    # Intentionally give identical error for both "no user" and "wrong password"
    # to prevent email enumeration attacks
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password.'}), 401

    user.update_streak()
    db.session.commit()

    # remember=True sets a persistent cookie (survives browser restart)
    from flask import session as _session
    _session.permanent = True
    login_user(user, remember=True)

    return jsonify({'user': user.to_dict()})

@app.route('/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Signed out successfully.'})

# ── User ──────────────────────────────────────────────────────────────────────

@app.route('/user/me')
@login_required
def me():
    return jsonify({'user': current_user.to_dict()})

@app.route('/user/xp', methods=['POST'])
@login_required
def add_xp():
    d   = request.get_json() or {}
    xp  = int(d.get('amount', 0))
    dur = int(d.get('duration_min', 0))

    current_user.xp           += xp
    current_user.quizzes_done += int(d.get('quiz_done', 0))
    current_user.courses_done += int(d.get('course_done', 0))
    current_user.total_time_min += dur
    current_user.update_rank()
    current_user.update_streak()

    if xp > 0:
        today = datetime.date.today()
        sess = StudySession.query.filter_by(user_id=current_user.id, date=today).first()
        if not sess:
            sess = StudySession(user_id=current_user.id, date=today)
            db.session.add(sess)
        sess.duration  += dur
        sess.xp_earned += xp
        existing_topics = json.loads(sess.topics or '[]')
        sess.topics = json.dumps(list(set(existing_topics + d.get('topics', []))))

    if d.get('quiz_done'):
        s = SystemStats.query.first()
        if s: s.total_quizzes += int(d.get('quiz_done', 0))
        # Track daily quiz activity for streak
        today_sess = get_or_create_today_session(current_user.id)
        today_sess.quizzes_done = (today_sess.quizzes_done or 0) + int(d.get('quiz_done', 0))

    db.session.commit()
    # Build daily status for response
    today = datetime.date.today()
    t_sess = StudySession.query.filter_by(user_id=current_user.id, date=today).first()
    secured = t_sess.streak_secured() if t_sess else False
    total_today = t_sess.activities_today() if t_sess else 0
    return jsonify({'user': current_user.to_dict(),
                    'daily_status': {
                        'total_today': total_today,
                        'streak_secured': secured,
                        'streak': current_user.streak,
                    }})

@app.route('/user/topics', methods=['POST'])
@login_required
def update_topics():
    d = request.get_json() or {}
    for item in d.get('results', []):
        upsert_topic(current_user.id, item['topic'], item['correct'], item['total'])
    return jsonify({'ok': True})

@app.route('/user/analytics')
@login_required
def analytics():
    uid    = current_user.id
    topics = TopicMastery.query.filter_by(user_id=uid)\
                .filter(TopicMastery.total > 0).order_by(TopicMastery.total.desc()).limit(12).all()
    thirty = datetime.date.today() - datetime.timedelta(days=30)
    sessions = StudySession.query.filter_by(user_id=uid)\
                .filter(StudySession.date >= thirty).all()
    heatmap  = {str(s.date): {'xp': s.xp_earned, 'min': s.duration} for s in sessions}
    due   = get_due_topics(uid)
    weak  = get_weak_topics(uid)
    avg   = round(sum(t.mastery_pct() for t in topics) / max(len(topics), 1)) if topics else 0
    return jsonify({'topics':      [t.to_dict() for t in topics],
                    'heatmap':     heatmap, 'due_count': len(due),
                    'weak_topics': weak, 'total_time_min': current_user.total_time_min,
                    'avg_score':   avg})

@app.route('/leaderboard')
@login_required
def leaderboard():
    top   = User.query.order_by(User.xp.desc()).limit(50).all()
    total = User.query.count()
    board = [{'rank': i+1, 'name': u.name, 'username': u.username, 'xp': u.xp,
              'rank_tier': u.rank, 'level': u.level, 'is_me': u.id == current_user.id}
             for i, u in enumerate(top)]
    return jsonify({'total_users': total, 'leaderboard': board})

# ── Saved Quizzes ─────────────────────────────────────────────────────────────

@app.route('/quiz/saved')
@login_required
def saved_quizzes():
    quizzes = SavedQuiz.query.filter_by(user_id=current_user.id)\
                .order_by(SavedQuiz.created_at.desc()).limit(20).all()
    return jsonify({'quizzes': [{'id': q.id, 'title': q.title,
        'questions': json.loads(q.questions), 'created_at': str(q.created_at)} for q in quizzes]})

@app.route('/quiz/save', methods=['POST'])
@login_required
def save_quiz():
    d  = request.get_json() or {}
    sq = SavedQuiz(user_id=current_user.id, title=d.get('title', 'Quiz'),
                   questions=json.dumps(d.get('questions', [])))
    db.session.add(sq)
    db.session.commit()
    return jsonify({'id': sq.id, 'message': 'Quiz saved for offline use.'})

# ── File Extraction ────────────────────────────────────────────────────────────

@app.route('/upload/extract', methods=['POST'])
@login_required
def upload_extract():
    files = request.files.getlist('files')
    if not files or not any(f.filename for f in files):
        return jsonify({'error': 'No files received — please select at least one file.'}), 400

    combined  = []
    filenames = []
    warnings  = []

    for f in files:
        if not f or not f.filename: continue
        try:
            text, err = proc_file(f)
            if err: warnings.append(f'{f.filename}: {err}')
            if text and text.strip():
                combined.append(f'=== {f.filename} ===\n{text.strip()}')
                filenames.append(f.filename)
        except Exception as e:
            warnings.append(f'{f.filename}: unexpected error — {e}')

    if not combined:
        detail = ' | '.join(warnings) if warnings else 'Unknown error.'
        return jsonify({'error': f'Could not extract text from any file. {detail}'}), 400

    s = SystemStats.query.first()
    if s: s.total_files += len(filenames)

    # Track daily file upload activity for streak
    today_sess = get_or_create_today_session(current_user.id)
    today_sess.files_uploaded = (today_sess.files_uploaded or 0) + len(filenames)
    db.session.commit()

    full = '\n\n'.join(combined)
    secured = today_sess.streak_secured()
    return jsonify({'text': full, 'char_count': len(full),
                    'filenames': filenames, 'warnings': warnings,
                    'daily_status': {
                        'total_today': today_sess.activities_today(),
                        'streak_secured': secured,
                        'streak': current_user.streak,
                    }})

@app.route('/ai/transcribe-audio', methods=['POST'])
@login_required
def transcribe_audio():
    """Accept audio file upload, return transcribed text for quiz/note generation."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file received.'}), 400
    f = request.files['audio']
    if not f or not f.filename:
        return jsonify({'error': 'Empty audio file.'}), 400
    data = f.read()
    text = ext_audio(data, f.filename)
    if not text.strip():
        return jsonify({'error': 'Could not transcribe audio — ensure file has clear speech and ffmpeg is installed.'}), 422
    return jsonify({'text': text, 'char_count': len(text)})


# ── AI Routes ─────────────────────────────────────────────────────────────────

@app.route('/ai/generate-quiz', methods=['POST'])
@login_required
def gen_quiz():
    d     = request.get_json() or {}
    text  = (d.get('text') or '').strip()
    title = (d.get('title') or 'Course').strip()
    nq    = min(30, max(5, int(d.get('num_questions', 10))))
    diff  = d.get('difficulty', 'intermediate')
    weak  = get_weak_topics(current_user.id)
    wh    = f'\nPrioritise these weak areas: {", ".join(weak)}' if weak else ''

    prompt = f"""Expert AI curriculum engine. Generate exactly {nq} multiple-choice questions.
Difficulty: {diff} | Topic: {title}{wh}
Material: {text[:4000] if text else f'Create comprehensive questions about: {title}'}
Rules: each question needs a "topic" field (2-4 word concept label). 4 plausible options, one correct.
Output ONLY valid JSON — no markdown, no backticks:
{{"questions":[{{"q":"Question text","opts":["A) o1","B) o2","C) o3","D) o4"],"ans":"A","exp":"Clear explanation.","topic":"Concept"}}]}}"""

    try:
        raw = cohere(prompt)
        raw = re.sub(r'```(?:json)?\n?|\n?```', '', raw).strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: raw = m.group(0)
        qs = json.loads(raw).get('questions', [])
        if not qs: raise ValueError('empty')
        s = SystemStats.query.first()
        if s: s.total_quizzes += nq; db.session.commit()
        return jsonify({'questions': qs})
    except (json.JSONDecodeError, ValueError):
        return jsonify({'questions': fallback_qs(title, nq), 'note': 'fallback'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai/adaptive-quiz', methods=['POST'])
@login_required
def adaptive_quiz():
    d     = request.get_json() or {}
    nq    = min(20, max(5, int(d.get('num_questions', 10))))
    weak  = get_weak_topics(current_user.id, 8)
    due   = [t.topic for t in get_due_topics(current_user.id)]
    focus = list(set(due + weak))
    if not focus:
        return jsonify({'error': 'Complete some quizzes first so we can personalise for you!'}), 400

    prompt = f"""Adaptive AI tutor. Generate exactly {nq} questions targeting: {', '.join(focus[:8])}
ONLY ask about these topics. Start easier, get progressively harder. Each needs a "topic" field.
Output ONLY valid JSON:
{{"questions":[{{"q":"Q","opts":["A) o1","B) o2","C) o3","D) o4"],"ans":"A","exp":"E.","topic":"T"}}]}}"""
    try:
        raw = cohere(prompt)
        raw = re.sub(r'```(?:json)?\n?|\n?```', '', raw).strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: raw = m.group(0)
        qs = json.loads(raw).get('questions', [])
        if not qs: raise ValueError('empty')
        return jsonify({'questions': qs, 'focus_topics': focus})
    except Exception:
        return jsonify({'questions': fallback_qs(', '.join(focus[:2]), nq), 'focus_topics': focus})

@app.route('/ai/explain-homework', methods=['POST'])
@login_required
def explain_hw():
    mode = request.form.get('mode', 'detailed')
    desc = (request.form.get('description') or '').strip()
    ocr_text = ''; ocr_err = None

    if 'image' in request.files:
        f = request.files['image']
        if f and f.filename:
            ocr_text, ocr_err = ocr_image(f, f.filename)

    combined = '\n'.join(filter(None, [ocr_text, desc])).strip()
    if not combined:
        msg = f'Image could not be read ({ocr_err}). Please type the problem below.' if ocr_err \
              else 'Please upload a photo of the problem OR type it in the text box.'
        return jsonify({'error': msg}), 400

    mode_map = {
        'simple':   "simple — Explain Like I'm 5, use everyday analogies, avoid jargon",
        'detailed': 'detailed and comprehensive with full working steps',
        'exam':     'exam-ready — include key formulas, common mistakes to avoid, and exam tips',
    }
    prompt = f"""You are an expert tutor. Explain this problem in a {mode_map.get(mode, 'detailed')} way.

Problem: {combined}

Use exactly these numbered sections:
1. Problem Identification — what type of problem is this?
2. Key Concepts / Formulas needed
3. Step-by-Step Solution — number each step clearly
4. Final Answer — state it clearly
5. Practice Problem — give one similar problem to try

Be clear, thorough, and educational."""

    try:
        result = cohere(prompt)
        # Track daily homework activity for streak
        today_sess = get_or_create_today_session(current_user.id)
        today_sess.hw_images = (today_sess.hw_images or 0) + 1
        db.session.commit()
        secured = today_sess.streak_secured()
        return jsonify({'explanation': result, 'ocr_text': ocr_text,
                        'daily_status': {
                            'total_today': today_sess.activities_today(),
                            'streak_secured': secured,
                            'streak': current_user.streak,
                        }})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai/check-solution', methods=['POST'])
@login_required
def check_solution():
    problem = (request.form.get('problem') or '').strip()
    answer  = (request.form.get('answer') or '').strip()
    ocr_text = ''

    if 'image' in request.files:
        f = request.files['image']
        if f and f.filename:
            ocr_text, _ = ocr_image(f, f.filename)

    student_work = '\n'.join(filter(None, [ocr_text, answer])).strip()
    if not student_work:
        return jsonify({'error': 'Please write your answer or upload a photo of your working.'}), 400

    prompt = f"""You are a strict but encouraging tutor reviewing a student's answer.

Problem: {problem or '(evaluate the student work on its own merits)'}
Student answer/working: {student_work}

Evaluate clearly:
1. Is the answer CORRECT or INCORRECT? (state this in the very first sentence)
2. If incorrect — what specific mistake did the student make?
3. What is the correct approach/answer?
4. Encouragement and one tip to avoid this mistake next time.

Be specific, constructive, and positive."""

    try:
        feedback   = cohere(prompt)
        is_correct = any(w in feedback.lower()[:120]
                         for w in ['correct', 'right', 'well done', 'excellent', 'perfect', 'great job'])
        # Track daily homework activity for streak
        today_sess = get_or_create_today_session(current_user.id)
        today_sess.hw_images = (today_sess.hw_images or 0) + 1
        db.session.commit()
        secured = today_sess.streak_secured()
        return jsonify({'feedback': feedback, 'is_correct': is_correct,
                        'daily_status': {
                            'total_today': today_sess.activities_today(),
                            'streak_secured': secured,
                            'streak': current_user.streak,
                        }})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai/generate-summary', methods=['POST'])
@login_required
def gen_summary():
    d     = request.get_json() or {}
    text  = (d.get('text') or '').strip()
    title = (d.get('title') or 'Content').strip()

    prompt = f"""Expert study assistant. Create a complete study package for:

Title: {title}
Content: {text[:5000] if text else f'Generate comprehensive study material about: {title}'}

Output ONLY valid JSON — no markdown, no backticks:
{{
  "summary": "3-5 paragraph summary",
  "key_points": ["Point 1","Point 2","...up to 15 points"],
  "formulas": ["Formula or key definition 1","..."],
  "mind_map": {{
    "center": "Main Topic",
    "branches": [
      {{"label":"Branch 1","children":["Sub-point A","Sub-point B"]}},
      {{"label":"Branch 2","children":["Sub-point C","Sub-point D"]}}
    ]
  }},
  "difficulty_areas": ["Concept typically hard to understand"]
}}"""

    try:
        raw = cohere(prompt, max_tok=3000)
        raw = re.sub(r'```(?:json)?\n?|\n?```', '', raw).strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: raw = m.group(0)
        return jsonify(json.loads(raw))
    except json.JSONDecodeError:
        return jsonify({'error': 'Could not parse AI response — please try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai/simulation', methods=['POST'])
@login_required
def gen_simulation():
    d       = request.get_json() or {}
    concept = (d.get('concept') or '').strip()
    if not concept:
        return jsonify({'error': 'Please enter a concept to simulate.'}), 400

    prompt = (f'Create a self-contained interactive HTML snippet (no external dependencies) '
              f'that visually demonstrates: "{concept}". '
              f'Use only HTML, CSS, and vanilla JavaScript. Must be interactive. '
              f'Max 80 lines. Wrap in <div class="sim-container">. Inline styles only. '
              f'Return ONLY the HTML snippet — no explanation, no markdown.')
    try:
        html = cohere(prompt, max_tok=2000)
        html = re.sub(r'<script[^>]*src=[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        return jsonify({'html': html})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai/mascot', methods=['POST'])
@login_required
def mascot():
    """Context-aware mascot messages. Local fallbacks first; Cohere only when context is rich."""
    d     = request.get_json(silent=True) or {}
    page  = d.get('page', 'dashboard')
    ctx   = d.get('context', {})
    first = current_user.name.split()[0]
    score = ctx.get('score')

    fallbacks = {
        'dashboard':    f"Hey {first}! {current_user.xp} XP and counting. Keep that streak alive!",
        'upload':       'Drop your file — I will build a quiz in seconds!',
        'quiz':         f"Think carefully before you tap, {first}!",
        'score':        f"{'Brilliant work!' if (score or 0)>=70 else 'Every quiz makes you sharper!'} Keep it going.",
        'homework':     'Snap the problem and I will explain every step clearly.',
        'analytics':    'Your weak spots today become your strengths tomorrow.',
        'leaderboard':  f"Keep grinding! The top spot is waiting for you, {first}.",
        'settings':     'Keep that streak alive! See you in tomorrow\'s session.',
        'summary':      'Mind map ready! Every branch is a key concept to master.',
    }

    has_rich_ctx = bool(score is not None or ctx.get('weakTopic') or ctx.get('file_title'))
    if has_rich_ctx and COHERE_API_KEY:
        parts = [f"user={first}", f"xp={current_user.xp}", f"rank={current_user.rank}",
                 f"streak={current_user.streak} days"]
        if score is not None:        parts.append(f"just_scored={score}%")
        if ctx.get('weakTopic'):     parts.append(f"weak_at={ctx['weakTopic']}")
        if ctx.get('file_title'):    parts.append(f"studying={ctx['file_title']}")
        try:
            msg = cohere(
                f"You are Brainy, cool mascot of Brainliant learning app. "
                f"Context: {', '.join(parts)}. Page: {page}. "
                f"Write ONE short encouraging sentence (max 18 words). No quotes, just the sentence.",
                max_tok=60
            ).strip().strip('"\'')
            if msg and len(msg) > 5:
                return jsonify({'message': msg[:160], 'source': 'ai'})
        except Exception:
            pass

    return jsonify({'message': fallbacks.get(page, f"Ready to learn something great, {first}!"),
                    'source': 'local'})

# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large — maximum upload size is 50 MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': f'Internal server error. Please try again.'}), 500

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"""
{'='*55}
  Brainliant — AI Learning Platform
  http://localhost:{port}
  Mode:   {'PRODUCTION' if IS_PRODUCTION else 'development'}
  DB:     {raw_db_url[:40]}{'...' if len(raw_db_url)>40 else ''}
  Cohere: {'✓ configured' if COHERE_API_KEY else '✗ MISSING'}
  OCR:    {'✓ configured' if OCR_API_KEY else '✗ MISSING'}
{'='*55}
""")
    app.run(debug=not IS_PRODUCTION, port=port, host='0.0.0.0')
