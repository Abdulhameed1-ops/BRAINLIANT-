import os, json, hashlib, uuid, datetime, re, math
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'brainliant-change-this-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///brainliant.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=30)

db = SQLAlchemy(app)

COHERE_API_KEY = os.getenv('COHERE_API_KEY', '')
COHERE_MODEL   = 'command-a-03-2025'
OCR_API_KEY    = os.getenv('OCR_API_KEY', '')

RANKS   = ['Beginner','Explorer','Scholar','Strategist','Master','Grandmaster','Legend']
RANK_XP = [0, 500, 1500, 3500, 7000, 12000, 20000]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def hash_pw(pw):   return hashlib.sha256(pw.encode()).hexdigest()
def check_pw(pw,h):return hashlib.sha256(pw.encode()).hexdigest() == h

# ─── MODELS ───────────────────────────────────────────────────────────────────

class User(db.Model):
    id            = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name          = db.Column(db.String(100), nullable=False)
    username      = db.Column(db.String(50),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(64),  nullable=False)
    xp            = db.Column(db.Integer, default=0)
    level         = db.Column(db.Integer, default=1)
    rank          = db.Column(db.String(50), default='Beginner')
    streak        = db.Column(db.Integer, default=1)
    last_active   = db.Column(db.Date, default=datetime.date.today)
    courses_done  = db.Column(db.Integer, default=0)
    quizzes_done  = db.Column(db.Integer, default=0)
    total_time_min= db.Column(db.Integer, default=0)  # total study minutes
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def update_rank(self):
        for i in range(len(RANK_XP)-1,-1,-1):
            if self.xp >= RANK_XP[i]: self.rank=RANKS[i]; self.level=i+1; break

    def update_streak(self):
        today = datetime.date.today()
        if self.last_active and (today - self.last_active).days == 1:
            self.streak += 1
        elif self.last_active and (today - self.last_active).days > 1:
            self.streak = 1
        self.last_active = today

    def rank_index(self):  return RANKS.index(self.rank) if self.rank in RANKS else 0
    def rank_progress(self):
        ri=self.rank_index(); lo=RANK_XP[ri]; hi=RANK_XP[ri+1] if ri+1<len(RANK_XP) else lo+5000
        return min(100,round((self.xp-lo)/max(hi-lo,1)*100))
    def global_position(self): return User.query.filter(User.xp>self.xp).count()+1
    def percentile(self):
        total=User.query.count(); pos=self.global_position()
        return max(0,round((1-pos/max(total,1))*100))
    def next_rank(self):
        ri=self.rank_index(); return RANKS[min(ri+1,len(RANKS)-1)]
    def next_rank_xp(self):
        ri=self.rank_index(); return RANK_XP[min(ri+1,len(RANK_XP)-1)]

    def to_dict(self):
        return {'id':self.id,'name':self.name,'username':self.username,'email':self.email,
                'xp':self.xp,'level':self.level,'rank':self.rank,'streak':self.streak,
                'courses':self.courses_done,'quizzes_done':self.quizzes_done,
                'total_time_min':self.total_time_min,
                'global_position':self.global_position(),'percentile':self.percentile(),
                'rank_progress':self.rank_progress(),'next_rank':self.next_rank(),
                'next_rank_xp':self.next_rank_xp()}

class TopicMastery(db.Model):
    """Tracks per-topic performance for adaptive learning."""
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    topic      = db.Column(db.String(200), nullable=False)
    correct    = db.Column(db.Integer, default=0)
    total      = db.Column(db.Integer, default=0)
    last_seen  = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    next_review= db.Column(db.DateTime, default=datetime.datetime.utcnow)  # spaced repetition
    interval   = db.Column(db.Integer, default=1)  # days until next review

    def mastery_pct(self):
        return round(self.correct/max(self.total,1)*100)

    def schedule_next_review(self, was_correct):
        """Simple SM-2 spaced repetition."""
        if was_correct:
            self.interval = min(self.interval*2, 30)
        else:
            self.interval = 1
        self.next_review = datetime.datetime.utcnow()+datetime.timedelta(days=self.interval)
        self.last_seen   = datetime.datetime.utcnow()

    def to_dict(self):
        return {'topic':self.topic,'correct':self.correct,'total':self.total,
                'mastery_pct':self.mastery_pct(),'last_seen':str(self.last_seen),
                'next_review':str(self.next_review),'interval':self.interval}

class StudySession(db.Model):
    """Records each study session for analytics/heatmap."""
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    date       = db.Column(db.Date, default=datetime.date.today)
    duration   = db.Column(db.Integer, default=0)   # minutes
    xp_earned  = db.Column(db.Integer, default=0)
    topics     = db.Column(db.Text, default='[]')   # JSON list of topic strings

class SavedQuiz(db.Model):
    """Stores generated quizzes for offline access."""
    id         = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    title      = db.Column(db.String(200), nullable=False)
    questions  = db.Column(db.Text, nullable=False)   # JSON
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class SystemStats(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    total_users   = db.Column(db.Integer, default=18294)
    total_quizzes = db.Column(db.Integer, default=342817)
    total_files   = db.Column(db.Integer, default=89421)

# ─── AUTH DECORATOR ────────────────────────────────────────────────────────────

def require_login(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args,**kwargs):
        uid=session.get('user_id')
        if not uid: return jsonify({'error':'Not authenticated'}),401
        user=User.query.get(uid)
        if not user: session.clear(); return jsonify({'error':'User not found'}),401
        request.current_user=user
        return f(*args,**kwargs)
    return wrapper

# ─── COHERE ─────────────────────────────────────────────────────────────────
# Uses v1/chat (same as reference app) — stable and widely tested

def call_cohere(prompt, max_tokens=4096):
    if not COHERE_API_KEY:
        raise ValueError('Cohere API key not set. Add COHERE_API_KEY to your .env file.')
    headers = {'Authorization':f'Bearer {COHERE_API_KEY}','Content-Type':'application/json'}
    payload  = {'model':COHERE_MODEL,'message':prompt,'max_tokens':max_tokens,'temperature':0.7}
    try:
        resp = requests.post('https://api.cohere.ai/v1/chat',
                             headers=headers, json=payload, timeout=90)
    except Exception as e:
        raise ValueError(f'Network error reaching Cohere: {e}')
    if not resp.ok:
        raise ValueError(f'Cohere {resp.status_code}: {resp.text[:300]}')
    data = resp.json()
    text = data.get('text','')
    if not text:
        raise ValueError('Cohere returned an empty response')
    return text

# ─── OCR ─────────────────────────────────────────────────────────────────────
# Multipart file upload (reference app pattern) — more reliable than base64

def extract_text_from_image(file_storage_or_bytes, filename='image.png'):
    """
    Accept a Flask FileStorage OR raw bytes.
    Returns (text_str, error_or_None).
    """
    import io as _io
    if not OCR_API_KEY:
        return '', 'OCR_API_KEY not set in .env'
    try:
        if hasattr(file_storage_or_bytes, 'read'):
            file_storage_or_bytes.seek(0)
            raw = file_storage_or_bytes.read()
            fname = getattr(file_storage_or_bytes,'filename',None) or filename
        else:
            raw = file_storage_or_bytes
            fname = filename
        # Normalise with PIL for better accuracy
        try:
            from PIL import Image
            img = Image.open(_io.BytesIO(raw)).convert('RGB')
            buf = _io.BytesIO(); img.save(buf, format='PNG')
            raw = buf.getvalue(); fname = 'image.png'
        except Exception:
            pass
        files   = {'file': (fname, raw, 'image/png')}
        payload = {'apikey':OCR_API_KEY,'language':'eng',
                   'isOverlayRequired':False,'detectOrientation':True,
                   'scale':True,'OCREngine':2}
        res = requests.post('https://api.ocr.space/parse/image',
                            data=payload, files=files, timeout=30)
        if res.status_code != 200:
            return '', f'OCR service returned {res.status_code}'
        data = res.json()
        if data.get('IsErroredOnProcessing'):
            msgs = data.get('ErrorMessage', ['Unknown OCR error'])
            return '', msgs[0] if isinstance(msgs,list) else str(msgs)
        parsed = data.get('ParsedResults', [])
        if not parsed:
            return '', 'No text detected in image'
        text = parsed[0].get('ParsedText','').strip()
        return (text, None) if text else ('', 'No readable text in image')
    except requests.exceptions.Timeout:
        return '', 'OCR timed out — try again'
    except Exception as e:
        return '', f'OCR error: {str(e)}'

# ─── FILE TEXT EXTRACTORS (no extra packages needed) ──────────────────────────

def extract_txt(data):
    """Plain text / markdown."""
    for enc in ('utf-8','latin-1','cp1252'):
        try: return data.decode(enc)
        except: pass
    return data.decode('utf-8', errors='ignore')

def extract_pdf(data):
    """Extract text from PDF. Uses PyPDF2 if installed, else pure-Python fallback."""
    # Try PyPDF2 first (best results)
    try:
        import PyPDF2, io
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try: parts.append(page.extract_text() or '')
            except: pass
        text = '\n'.join(parts).strip()
        if len(text) > 50:
            return text
    except ImportError:
        pass
    except Exception:
        pass

    # Pure-Python fallback — extract text from PDF content streams
    import zlib
    text_parts = []
    stream_re = re.compile(rb'stream\r?\n(.*?)\r?\nendstream', re.DOTALL)
    for m in stream_re.finditer(data):
        chunk = m.group(1)
        try: chunk = zlib.decompress(chunk)
        except: pass
        decoded = chunk.decode('latin-1', errors='ignore')
        parts = re.findall(r'\(([^)\\]*(?:\\.[^)\\]*)*)\)\s*Tj', decoded)
        tj_parts = re.findall(r'\[([^\]]*)\]\s*TJ', decoded)
        for p in parts:
            text_parts.append(p.replace('\\n',' ').replace('\\r',' '))
        for p in tj_parts:
            inner = re.findall(r'\(([^)\\]*(?:\\.[^)\\]*)*)\)', p)
            text_parts.extend(inner)

    result = ' '.join(text_parts)
    result = re.sub(r'\\(\d{3})', lambda m: chr(int(m.group(1),8)), result)
    result = re.sub(r'\s+', ' ', result.replace('\\n',' ').replace('\\',' ')).strip()
    return result

def extract_docx(data):
    """Extract text from DOCX. Uses python-docx if installed, else ZIP+XML fallback."""
    # Try python-docx first (best results)
    try:
        import docx, io
        doc = docx.Document(io.BytesIO(data))
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        pass
    except Exception:
        pass

    # Pure-Python fallback — DOCX is just a ZIP of XML files
    import zipfile, io
    try:
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open('word/document.xml') as f:
                tree = ET.parse(f)
                ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                texts = [elem.text for elem in tree.getroot().iter(ns+'t') if elem.text]
                return ' '.join(texts)
    except Exception:
        return ''

def extract_file_text(filename, data):
    """Route to the correct extractor based on file extension."""
    fn = filename.lower()
    if fn.endswith('.pdf'):
        text = extract_pdf(data)
        if len(text.strip()) < 50:
            text = (f'PDF file: {filename}. The PDF could not be fully parsed. '
                    f'Please describe the topic in the title field so AI can generate relevant questions.')
        return text
    elif fn.endswith('.docx'):
        text = extract_docx(data)
        if len(text.strip()) < 10:
            text = f'Word document: {filename}'
        return text
    else:
        return extract_txt(data)

# ─── ADAPTIVE HELPERS ─────────────────────────────────────────────────────────

def get_weak_topics(user_id, limit=5):
    topics = TopicMastery.query.filter_by(user_id=user_id)\
        .filter(TopicMastery.total>0)\
        .order_by(TopicMastery.correct/TopicMastery.total).limit(limit).all()
    return [t.topic for t in topics]

def get_due_topics(user_id):
    now=datetime.datetime.utcnow()
    return TopicMastery.query.filter_by(user_id=user_id)\
        .filter(TopicMastery.next_review<=now).all()

def upsert_topic(user_id, topic, correct, total):
    tm=TopicMastery.query.filter_by(user_id=user_id,topic=topic).first()
    if not tm:
        tm=TopicMastery(user_id=user_id,topic=topic,correct=0,total=0)
        db.session.add(tm)
    tm.correct+=correct; tm.total+=total
    tm.schedule_next_review(correct>0)
    db.session.commit()

def fallback_quiz(topic, n):
    base=[
        {'q':f'What is the foundational concept behind "{topic}"?',
         'opts':['A) Core framework','B) Minor detail','C) Unrelated tangent','D) Deprecated idea'],
         'ans':'A','exp':'The foundational framework underpins all related knowledge and should be understood first.','topic':topic},
        {'q':f'Which approach is most effective when studying "{topic}"?',
         'opts':['A) Passive re-reading','B) Active recall and practice','C) Memorising only','D) Skipping hard parts'],
         'ans':'B','exp':'Active recall (the testing effect) builds far stronger memory traces than passive review.','topic':topic},
        {'q':f'How should you handle difficult sections in "{topic}"?',
         'opts':['A) Skip entirely','B) Break into smaller chunks','C) Read faster','D) Only read summaries'],
         'ans':'B','exp':'Chunking reduces cognitive load and makes complex material manageable step by step.','topic':topic},
    ]
    return [dict(base[i%len(base)],q=f'Q{i+1}: '+base[i%len(base)]['q']) for i in range(n)]

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index(): return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

# ── File extraction endpoint ───────────────────────────────────────────────────

@app.route('/upload/extract', methods=['POST'])
@require_login
def upload_extract():
    """
    Accept one or more uploaded files, extract their text server-side,
    and return the combined text to the frontend.
    """
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files received'}), 400

    combined = []
    filenames = []
    for f in files:
        try:
            data = f.read()
            text = extract_file_text(f.filename, data)
            if text:
                combined.append(f'=== {f.filename} ===\n{text}')
            filenames.append(f.filename)
        except Exception as e:
            combined.append(f'[Could not read {f.filename}: {str(e)}]')

    stats = SystemStats.query.first()
    if stats:
        stats.total_files += len(files)
        db.session.commit()

    full_text = '\n\n'.join(combined)
    return jsonify({
        'text': full_text,
        'char_count': len(full_text),
        'filenames': filenames
    })

# ── Auth ─────────────────────────────────────────────────────────────────────

@app.route('/auth/check')
def auth_check():
    uid=session.get('user_id')
    if not uid: return jsonify({'authenticated':False})
    user=User.query.get(uid)
    if not user: session.clear(); return jsonify({'authenticated':False})
    user.update_streak(); db.session.commit()
    return jsonify({'authenticated':True,'user':user.to_dict()})

@app.route('/auth/register',methods=['POST'])
def register():
    d=request.get_json()
    name=(d.get('name')or'').strip(); username=(d.get('username')or'').strip().lower()
    email=(d.get('email')or'').strip().lower(); password=(d.get('password')or'')
    if not all([name,username,email,password]): return jsonify({'error':'All fields required'}),400
    if len(password)<8: return jsonify({'error':'Password must be at least 8 characters'}),400
    if User.query.filter_by(email=email).first(): return jsonify({'error':'Email already registered'}),409
    if User.query.filter_by(username=username).first(): return jsonify({'error':'Username already taken'}),409
    user=User(name=name,username=username,email=email,password_hash=hash_pw(password))
    db.session.add(user)
    stats=SystemStats.query.first()
    if stats: stats.total_users+=1
    db.session.commit()
    session.permanent=True; session['user_id']=user.id
    return jsonify({'user':user.to_dict(),'message':'Account created'}),201

@app.route('/auth/login',methods=['POST'])
def login():
    d=request.get_json()
    email=(d.get('email')or'').strip().lower(); password=(d.get('password')or'')
    user=User.query.filter_by(email=email).first()
    if not user or not check_pw(password,user.password_hash):
        return jsonify({'error':'Invalid email or password'}),401
    user.update_streak(); db.session.commit()
    session.permanent=True; session['user_id']=user.id
    return jsonify({'user':user.to_dict()})

@app.route('/auth/logout',methods=['POST'])
def logout(): session.clear(); return jsonify({'message':'Logged out'})

# ── User ──────────────────────────────────────────────────────────────────────

@app.route('/user/me')
@require_login
def get_me(): return jsonify({'user':request.current_user.to_dict()})

@app.route('/user/xp',methods=['POST'])
@require_login
def add_xp():
    d=request.get_json(); user=request.current_user
    xp_amt=int(d.get('amount',0)); duration=int(d.get('duration_min',0))
    user.xp+=xp_amt; user.quizzes_done+=int(d.get('quiz_done',0))
    user.courses_done+=int(d.get('course_done',0)); user.total_time_min+=duration
    user.update_rank(); user.update_streak()
    # Log study session
    if xp_amt>0:
        today=datetime.date.today()
        sess=StudySession.query.filter_by(user_id=user.id,date=today).first()
        if not sess: sess=StudySession(user_id=user.id,date=today); db.session.add(sess)
        sess.duration+=duration; sess.xp_earned+=xp_amt
        topics=d.get('topics',[])
        existing=json.loads(sess.topics or '[]')
        sess.topics=json.dumps(list(set(existing+topics)))
    db.session.commit()
    stats=SystemStats.query.first()
    if stats and d.get('quiz_done'): stats.total_quizzes+=int(d.get('quiz_done',0)); db.session.commit()
    return jsonify({'user':user.to_dict()})

@app.route('/user/topics',methods=['POST'])
@require_login
def update_topics():
    d=request.get_json(); user=request.current_user
    for item in d.get('results',[]):
        upsert_topic(user.id,item['topic'],item['correct'],item['total'])
    return jsonify({'ok':True})

# ── Analytics ─────────────────────────────────────────────────────────────────

@app.route('/user/analytics')
@require_login
def analytics():
    user=request.current_user
    # Topic mastery
    topics=TopicMastery.query.filter_by(user_id=user.id)\
        .order_by(TopicMastery.total.desc()).limit(12).all()
    # 30-day activity heatmap
    thirty=datetime.date.today()-datetime.timedelta(days=30)
    sessions=StudySession.query.filter_by(user_id=user.id)\
        .filter(StudySession.date>=thirty).all()
    heatmap={str(s.date):{'xp':s.xp_earned,'min':s.duration} for s in sessions}
    # Due for review (spaced repetition)
    due=get_due_topics(user.id)
    weak=get_weak_topics(user.id)
    return jsonify({
        'topics':[t.to_dict() for t in topics],
        'heatmap':heatmap,
        'due_count':len(due),
        'weak_topics':weak,
        'total_time_min':user.total_time_min,
        'avg_score': round(sum(t.mastery_pct() for t in topics)/max(len(topics),1)) if topics else 0
    })

# ── Leaderboard ───────────────────────────────────────────────────────────────

@app.route('/leaderboard')
@require_login
def leaderboard():
    top=User.query.order_by(User.xp.desc()).limit(50).all()
    total=User.query.count(); me_id=request.current_user.id
    return jsonify({'total_users':total,'leaderboard':[
        {'rank':i+1,'name':u.name,'username':u.username,'xp':u.xp,
         'rank_tier':u.rank,'level':u.level,'is_me':u.id==me_id}
        for i,u in enumerate(top)]})

@app.route('/stats')
def get_stats():
    s=SystemStats.query.first()
    return jsonify({'users':s.total_users if s else 0,
                    'quizzes':s.total_quizzes if s else 0,'files':s.total_files if s else 0})

# ── Saved Quizzes (Offline) ────────────────────────────────────────────────────

@app.route('/quiz/saved')
@require_login
def get_saved_quizzes():
    quizzes=SavedQuiz.query.filter_by(user_id=request.current_user.id)\
        .order_by(SavedQuiz.created_at.desc()).limit(20).all()
    return jsonify({'quizzes':[{'id':q.id,'title':q.title,
        'questions':json.loads(q.questions),'created_at':str(q.created_at)} for q in quizzes]})

@app.route('/quiz/save',methods=['POST'])
@require_login
def save_quiz():
    d=request.get_json()
    sq=SavedQuiz(user_id=request.current_user.id,title=d.get('title','Quiz'),
                 questions=json.dumps(d.get('questions',[])))
    db.session.add(sq); db.session.commit()
    return jsonify({'id':sq.id,'message':'Quiz saved for offline use'})

# ── AI Routes ─────────────────────────────────────────────────────────────────

@app.route('/ai/generate-quiz',methods=['POST'])
@require_login
def generate_quiz():
    d=request.get_json()
    text=(d.get('text')or'').strip(); title=(d.get('title')or'Course').strip()
    num_q=min(30,max(5,int(d.get('num_questions',10)))); diff=d.get('difficulty','intermediate')
    # Adaptive: inject weak topics to bias questions
    weak=get_weak_topics(request.current_user.id)
    weak_hint=f'\nPrioritise questions on these weak areas the student struggles with: {", ".join(weak)}' if weak else ''

    prompt=f"""You are an expert AI curriculum engine. Generate exactly {num_q} quiz questions.

Difficulty: {diff}
Topic: {title}{weak_hint}
Material: {text[:4000] if text else f'Create comprehensive questions about: {title}'}

Rules:
- Each question must include a "topic" field (2-4 word concept label, e.g. "Quadratic Equations")
- Cover all major concepts progressively, harder questions last
- All 4 options must be plausible; only one correct

Output ONLY valid JSON, no markdown:
{{"questions":[{{"q":"Question text","opts":["A) option1","B) option2","C) option3","D) option4"],"ans":"A","exp":"2-3 sentence explanation.","topic":"Concept Name"}}]}}"""

    try:
        raw=call_cohere(prompt); raw=re.sub(r'```(?:json)?\n?|\n?```','',raw).strip()
        quiz=json.loads(raw); qs=quiz.get('questions',[])
        if not qs: raise ValueError('Empty')
        stats=SystemStats.query.first()
        if stats: stats.total_files+=1; stats.total_quizzes+=num_q; db.session.commit()
        return jsonify({'questions':qs})
    except (json.JSONDecodeError,ValueError):
        return jsonify({'questions':fallback_quiz(title,num_q),'note':'fallback'})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/ai/adaptive-quiz',methods=['POST'])
@require_login
def adaptive_quiz():
    """Generate a quiz targeting the user's weak/due topics."""
    d=request.get_json(); num_q=min(20,max(5,int(d.get('num_questions',10))))
    user=request.current_user
    weak=get_weak_topics(user.id,8)
    due=[t.topic for t in get_due_topics(user.id)]
    focus=list(set(due+weak))
    if not focus: return jsonify({'error':'Complete some quizzes first so we can personalise for you!'}),400

    prompt=f"""You are an adaptive AI tutor. Generate exactly {num_q} targeted quiz questions.

The student needs practice on these specific topics: {', '.join(focus[:8])}

Rules:
- ONLY ask about the listed topics above
- Start easier, get progressively harder
- Each question must include a "topic" field matching one from the list

Output ONLY valid JSON, no markdown:
{{"questions":[{{"q":"Question","opts":["A) o1","B) o2","C) o3","D) o4"],"ans":"A","exp":"Explanation.","topic":"Topic Name"}}]}}"""

    try:
        raw=call_cohere(prompt); raw=re.sub(r'```(?:json)?\n?|\n?```','',raw).strip()
        quiz=json.loads(raw); qs=quiz.get('questions',[])
        if not qs: raise ValueError('Empty')
        return jsonify({'questions':qs,'focus_topics':focus})
    except Exception:
        return jsonify({'questions':fallback_quiz(', '.join(focus[:2]),num_q),'focus_topics':focus})

@app.route('/ai/explain-homework',methods=['POST'])
@require_login
def explain_homework():
    mode=request.form.get('mode','detailed'); desc=(request.form.get('description')or'').strip()
    ocr_text=''
    if 'image' in request.files:
        f=request.files['image']
        if f and f.filename:
            ocr_text,ocr_err=extract_text_from_image(f,f.filename)
        else:
            ocr_text,ocr_err='',None
    else:
        ocr_err=None
    combined='\n'.join(filter(None,[ocr_text,desc])).strip()
    if not combined:
        msg='Please upload a photo or describe the problem in the box.'
        if ocr_err: msg=f'Image read failed: {ocr_err}. Please describe the problem instead.'
        return jsonify({'error':msg}),400
    mode_map={'simple':'simple and easy to understand (Explain Like I\'m 5 — use analogies, avoid jargon)',
              'detailed':'detailed and comprehensive with full working steps',
              'exam':'exam-ready with key formulas, common mistakes to avoid, and exam tips'}
    prompt=f"""You are an expert tutor. Explain this problem in a {mode_map.get(mode,'detailed')} way.

Problem: {combined}

Structure your response with these exact sections:
1. Problem Identification — what type of problem is this?
2. Key Concepts — what concepts or formulas are needed?
3. Step-by-Step Solution — number each step clearly
4. Final Answer — state the answer clearly
5. Practice Problem — give one similar problem to try

Be clear, educational, and thorough."""
    try:
        return jsonify({'explanation':call_cohere(prompt),'ocr_text':ocr_text})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/ai/check-solution',methods=['POST'])
@require_login
def check_solution():
    """Student uploads their answer — AI checks if it's correct."""
    problem=(request.form.get('problem')or'').strip()
    student_answer=(request.form.get('answer')or'').strip()
    ocr_text=''
    if 'image' in request.files:
        f=request.files['image']
        if f and f.filename:
            ocr_text,_=extract_text_from_image(f,f.filename)
        else:
            ocr_text=''
    student_work='\n'.join(filter(None,[ocr_text,student_answer])).strip()
    if not student_work: return jsonify({'error':'Please write your answer or upload a photo of your work.'}),400
    prompt=f"""You are a strict but encouraging tutor reviewing a student's answer.

Original Problem: {problem or 'unknown problem'}
Student's Answer/Work: {student_work}

Evaluate the student's work:
1. Is the final answer CORRECT or INCORRECT? (state this clearly first)
2. If incorrect, what specific mistake did the student make?
3. What is the correct approach?
4. Encouragement and tip to avoid this mistake next time.

Be specific, constructive, and positive in tone."""
    try:
        feedback=call_cohere(prompt)
        is_correct=any(w in feedback.lower()[:100] for w in ['correct','right','well done','excellent','perfect'])
        return jsonify({'feedback':feedback,'is_correct':is_correct})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/ai/generate-summary',methods=['POST'])
@require_login
def generate_summary():
    """Generate summary notes, key points, and mind map from uploaded content."""
    d=request.get_json(); text=(d.get('text')or'').strip(); title=(d.get('title')or'Content').strip()

    prompt=f"""You are an expert study assistant. Analyse this content and produce a complete study package.

Title: {title}
Content: {text[:5000] if text else f'Generate study material about: {title}'}

Output ONLY valid JSON, no markdown:
{{
  "summary": "3-5 paragraph summary of the entire content",
  "key_points": ["Point 1", "Point 2", "Point 3", "...up to 15 key points"],
  "formulas": ["Formula or definition 1", "..."],
  "mind_map": {{
    "center": "Main Topic",
    "branches": [
      {{"label": "Branch 1", "children": ["Sub-point A", "Sub-point B"]}},
      {{"label": "Branch 2", "children": ["Sub-point C", "Sub-point D"]}}
    ]
  }},
  "difficulty_areas": ["Concept that is typically hard to understand", "..."]
}}"""

    try:
        raw=call_cohere(prompt,max_tokens=3000); raw=re.sub(r'```(?:json)?\n?|\n?```','',raw).strip()
        data=json.loads(raw); return jsonify(data)
    except json.JSONDecodeError:
        return jsonify({'error':'Could not parse AI response. Try again.'}),500
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/ai/simulation',methods=['POST'])
@require_login
def generate_simulation():
    """AI generates an interactive SVG/HTML simulation for a concept."""
    d=request.get_json(); concept=(d.get('concept')or'').strip()
    prompt=f"""You are an expert at creating interactive educational HTML visualisations.

Create a self-contained interactive HTML snippet (no external dependencies) that visually demonstrates: "{concept}"

Requirements:
- Use only HTML, CSS, and vanilla JavaScript
- Must be interactive (clickable, draggable, or animated elements)
- Include a brief text label explaining what's shown
- Max 80 lines
- Wrap everything in a single <div class="sim-container"> element
- Use inline styles only
- Make it visually clear and educational

Return ONLY the HTML snippet, nothing else."""
    try:
        html=call_cohere(prompt,max_tokens=2000)
        # Sanitise — strip any script tags that load external resources
        html=re.sub(r'<script[^>]*src=[^>]*>.*?</script>','',html,flags=re.DOTALL)
        return jsonify({'html':html})
    except Exception as e:
        return jsonify({'error':str(e)}),500

# ─── DB INIT ──────────────────────────────────────────────────────────────────

@app.route('/ai/mascot',methods=['POST'])
@require_login
def mascot_say():
    """Short contextual message from Brainy based on which page user is on.
    Uses local fallbacks first; only calls Cohere if context is rich enough."""
    d=request.get_json(silent=True) or {}
    page=d.get('page','dashboard')
    ctx=d.get('context',{})
    user=request.current_user
    first=user.name.split()[0]

    # Always-available local fallbacks — no API needed
    fallbacks={
        'landing':   f"Welcome! I am Brainy. Tap Start to begin your journey.",
        'signup':    f"Creating your account! You are one step away from greatness.",
        'signin':    f"Welcome back! Your brain missed this.",
        'dashboard': f"Hey {first}! {user.xp} XP so far. Keep that streak alive!",
        'upload':    f"Drop your file here and I will cook up a quiz in seconds!",
        'quiz':      f"Question {ctx.get('qNum','')} of {ctx.get('total','')}. You got this, {first}!",
        'score':     f"You scored {ctx.get('score','')}%! {'Brilliant!' if (ctx.get('score') or 0)>=70 else 'Keep pushing!'}",
        'homework':  f"Snap that problem and I will explain every step clearly.",
        'analytics': f"Your weakest topic is {ctx.get('weakTopic','unknown')}. Let us fix that today!",
        'leaderboard':f"You are ranked #{user.global_position()} globally. Top 10 is coming!",
        'summary':   f"Mind map ready! Tap any branch to explore, {first}.",
        'settings':  f"All good here! Keep your streak going tomorrow.",
    }

    # Decide whether to call Cohere (only for richer contexts)
    use_ai = bool(ctx.get('score') or ctx.get('weakTopic') or ctx.get('file_title')) and bool(COHERE_API_KEY)

    if use_ai:
        ctx_info=[]
        if ctx.get('score') is not None: ctx_info.append(f"just scored {ctx['score']}%")
        if ctx.get('weakTopic'): ctx_info.append(f"weak at {ctx['weakTopic']}")
        if ctx.get('file_title'): ctx_info.append(f"studying '{ctx['file_title']}'")
        prompt=(f"You are Brainy, a cool encouraging mascot for Brainliant app. "
                f"User: {first}, XP: {user.xp}, Rank: {user.rank}, Streak: {user.streak} days. "
                f"Context: {', '.join(ctx_info)}. Page: {page}. "
                f"Write ONE short encouraging sentence (max 18 words). No quotes, no emojis, just the sentence.")
        try:
            msg=call_cohere(prompt,max_tokens=50).strip().strip('"\' ')
            if msg and len(msg)>5:
                return jsonify({'message':msg[:160],'source':'ai'})
        except Exception:
            pass

    return jsonify({'message':fallbacks.get(page,f"Ready to learn, {first}!"),'source':'local'})

# ─── DB INIT ──────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()
    if not SystemStats.query.first():
        db.session.add(SystemStats()); db.session.commit()

if __name__ == '__main__': app.run(debug=True,port=5000)
