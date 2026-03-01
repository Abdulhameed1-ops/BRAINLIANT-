"""
Brainliant v6 - zero extra deps beyond: flask requests Pillow python-docx openpyxl
Uses plain sqlite3, no SQLAlchemy, no PyPDF2.
"""
import os, json, hashlib, uuid, re, io, zlib, sqlite3, datetime
from functools import wraps
from flask import Flask, request, jsonify, render_template, session, send_from_directory
import requests as http_req

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'brainliant-dev-2025')
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=30)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SESSION_COOKIE_SECURE']   = False   # allow HTTP localhost
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

COHERE_API_KEY = os.environ.get('COHERE_API_KEY', '')
OCR_API_KEY    = os.environ.get('OCR_API_KEY', '')
DB_PATH        = os.environ.get('DB_PATH', 'brainliant.db')
RANKS          = ['Beginner','Explorer','Scholar','Strategist','Master','Grandmaster','Legend']
RANK_XP        = [0,500,1500,3500,7000,12000,20000]

def get_db():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL'); return c

def init_db():
    with get_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,name TEXT,username TEXT UNIQUE,
          email TEXT UNIQUE,password_hash TEXT,xp INTEGER DEFAULT 0,level INTEGER DEFAULT 1,
          rank TEXT DEFAULT 'Beginner',streak INTEGER DEFAULT 1,last_active TEXT DEFAULT '',
          courses_done INTEGER DEFAULT 0,quizzes_done INTEGER DEFAULT 0,
          total_time_min INTEGER DEFAULT 0,created_at TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS topic_mastery(id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT,topic TEXT,correct INTEGER DEFAULT 0,total INTEGER DEFAULT 0,
          last_seen TEXT DEFAULT '',next_review TEXT DEFAULT '',interval_days INTEGER DEFAULT 1,
          UNIQUE(user_id,topic));
        CREATE TABLE IF NOT EXISTS study_sessions(id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT,date TEXT,duration INTEGER DEFAULT 0,xp_earned INTEGER DEFAULT 0,
          topics TEXT DEFAULT '[]',UNIQUE(user_id,date));
        CREATE TABLE IF NOT EXISTS saved_quizzes(id TEXT PRIMARY KEY,user_id TEXT,
          title TEXT,questions TEXT,created_at TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS system_stats(id INTEGER PRIMARY KEY,
          total_users INTEGER DEFAULT 18294,total_quizzes INTEGER DEFAULT 342817,
          total_files INTEGER DEFAULT 89421);
        INSERT OR IGNORE INTO system_stats(id) VALUES(1);
        """)

init_db()

def q1(sql,p=()):
    with get_db() as c: r=c.execute(sql,p).fetchone(); return dict(r) if r else None
def qa(sql,p=()):
    with get_db() as c: return [dict(r) for r in c.execute(sql,p).fetchall()]
def qx(sql,p=()):
    with get_db() as c: c.execute(sql,p); c.commit()

def hash_pw(pw):    return hashlib.sha256(pw.encode()).hexdigest()
def chk_pw(pw,h):   return hashlib.sha256(pw.encode()).hexdigest()==h

def rank_for(xp):
    for i in range(len(RANK_XP)-1,-1,-1):
        if xp>=RANK_XP[i]: return RANKS[i],i+1
    return RANKS[0],1

def u2d(u):
    rank,level=rank_for(u['xp'])
    ri=RANKS.index(rank); lo=RANK_XP[ri]; hi=RANK_XP[ri+1] if ri+1<len(RANK_XP) else lo+5000
    prog=min(100,round((u['xp']-lo)/max(hi-lo,1)*100))
    total=q1("SELECT COUNT(*) n FROM users")['n']
    pos=q1("SELECT COUNT(*) n FROM users WHERE xp>?",(u['xp'],))['n']+1
    pct=max(0,round((1-pos/max(total,1))*100))
    return {'id':u['id'],'name':u['name'],'username':u['username'],'email':u['email'],
            'xp':u['xp'],'level':level,'rank':rank,'streak':u['streak'],
            'courses':u['courses_done'],'quizzes_done':u['quizzes_done'],
            'total_time_min':u['total_time_min'],'global_position':pos,'percentile':pct,
            'rank_progress':prog,'next_rank':RANKS[min(ri+1,len(RANKS)-1)],
            'next_rank_xp':RANK_XP[min(ri+1,len(RANK_XP)-1)]}

def upd_streak(uid,today_str):
    u=q1("SELECT last_active,streak FROM users WHERE id=?",(uid,))
    if not u: return
    streak=u['streak'] or 1
    if u['last_active']:
        try:
            diff=(datetime.datetime.fromisoformat(today_str).date()-
                  datetime.datetime.fromisoformat(u['last_active']).date()).days
            if diff==1: streak+=1
            elif diff>1: streak=1
        except: pass
    qx("UPDATE users SET streak=?,last_active=? WHERE id=?",(streak,today_str,uid))

def require_login(f):
    @wraps(f)
    def w(*a,**kw):
        uid=session.get('user_id')
        if not uid: return jsonify({'error':'Not authenticated — please log in'}),401
        u=q1("SELECT * FROM users WHERE id=?",(uid,))
        if not u: session.clear(); return jsonify({'error':'Session expired — please log in again'}),401
        request.cu=u; return f(*a,**kw)
    return w

# ── COHERE ────────────────────────────────────────────────────────────────────
def cohere(prompt, max_tok=4096):
    if not COHERE_API_KEY:
        raise ValueError('COHERE_API_KEY missing. Create .env with: COHERE_API_KEY=your_key')
    try:
        r=http_req.post('https://api.cohere.ai/v1/chat',
            headers={'Authorization':f'Bearer {COHERE_API_KEY}','Content-Type':'application/json'},
            json={'model':'command-a-03-2025','message':prompt,'max_tokens':max_tok,'temperature':0.7},
            timeout=90)
    except http_req.exceptions.ConnectionError:
        raise ValueError('Cannot reach Cohere AI — check your internet connection')
    except http_req.exceptions.Timeout:
        raise ValueError('Cohere AI timed out — please try again')
    if r.status_code==401: raise ValueError('Invalid Cohere API key')
    if r.status_code==429: raise ValueError('Cohere rate limit — wait a moment then try again')
    if not r.ok: raise ValueError(f'Cohere error {r.status_code}: {r.text[:200]}')
    t=r.json().get('text','')
    if not t: raise ValueError('Cohere returned empty response — try again')
    return t

# ── OCR ───────────────────────────────────────────────────────────────────────
def ocr_image(file_obj, fname='image.png'):
    if not OCR_API_KEY: return '','OCR_API_KEY not set in .env'
    try:
        if hasattr(file_obj,'seek'): file_obj.seek(0)
        raw=file_obj.read() if hasattr(file_obj,'read') else file_obj
        try:
            from PIL import Image
            img=Image.open(io.BytesIO(raw)).convert('RGB')
            buf=io.BytesIO(); img.save(buf,format='PNG'); raw=buf.getvalue(); fname='image.png'
        except: pass
        r=http_req.post('https://api.ocr.space/parse/image',
            data={'apikey':OCR_API_KEY,'language':'eng','isOverlayRequired':False,
                  'detectOrientation':True,'scale':True,'OCREngine':2},
            files={'file':(fname,raw,'image/png')},timeout=30)
        if r.status_code!=200: return '',f'OCR returned {r.status_code}'
        d=r.json()
        if d.get('IsErroredOnProcessing'):
            msgs=d.get('ErrorMessage',['OCR error'])
            return '',msgs[0] if isinstance(msgs,list) else str(msgs)
        parsed=d.get('ParsedResults',[])
        if not parsed: return '','No text detected'
        t=parsed[0].get('ParsedText','').strip()
        return (t,None) if t else ('','No readable text in image')
    except http_req.exceptions.Timeout: return '','OCR timed out'
    except Exception as e: return '',f'OCR error: {e}'

# ── FILE EXTRACTORS ───────────────────────────────────────────────────────────
def ext_pdf(data):
    parts=[]
    for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream',data,re.DOTALL):
        chunk=m.group(1)
        try: chunk=zlib.decompress(chunk)
        except: pass
        dec=chunk.decode('latin-1',errors='ignore')
        for p in re.findall(r'\(([^)\\]*(?:\\.[^)\\]*)*)\)\s*Tj',dec): parts.append(p.replace('\\n',' '))
        for b in re.findall(r'\[([^\]]*)\]\s*TJ',dec):
            parts.extend(re.findall(r'\(([^)\\]*(?:\\.[^)\\]*)*)\)',b))
    r=re.sub(r'\s+',' ',' '.join(parts).replace('\\n',' ').replace('\\',' ')).strip()
    return r

def ext_docx(data):
    try:
        import docx; doc=docx.Document(io.BytesIO(data))
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    except: pass
    try:
        import zipfile; from xml.etree import ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open('word/document.xml') as f:
                ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                return ' '.join(e.text for e in ET.parse(f).getroot().iter(ns+'t') if e.text)
    except: return ''

def ext_xlsx(data):
    try:
        import openpyxl; wb=openpyxl.load_workbook(io.BytesIO(data)); parts=[]
        for sh in wb.worksheets:
            parts.append(f'[Sheet:{sh.title}]')
            for row in sh.iter_rows(values_only=True):
                t=' | '.join(str(c) for c in row if c is not None)
                if t.strip(): parts.append(t)
        return '\n'.join(parts)
    except: return ''

def proc_file(fs):
    fn=(fs.filename or '').lower(); data=fs.read()
    if fn.endswith('.pdf'):
        t=ext_pdf(data)
        if len(t.strip())<30: return None,'PDF text unreadable (may be scanned). Enter topic in Title field.'
        return t,None
    if fn.endswith(('.docx','.doc')):
        t=ext_docx(data); return (t,None) if t.strip() else (None,'Could not read DOCX')
    if fn.endswith(('.xlsx','.xls')):
        t=ext_xlsx(data); return (t,None) if t.strip() else (None,'Could not read Excel')
    if fn.endswith(('.png','.jpg','.jpeg','.gif','.bmp','.tiff','.webp')):
        t,err=ocr_image(io.BytesIO(data),fs.filename)
        return (t,err) if not (err and not t) else (None,f'OCR failed: {err}')
    if fn.endswith(('.txt','.md','.csv')):
        for enc in ('utf-8','latin-1','cp1252'):
            try: return data.decode(enc),None
            except: pass
        return data.decode('utf-8',errors='ignore'),None
    return None,f'Unsupported type: {fn.rsplit(".",1)[-1]}'

# ── ADAPTIVE ─────────────────────────────────────────────────────────────────
def weak_topics(uid,n=5):
    rows=qa("SELECT topic,correct,total FROM topic_mastery WHERE user_id=? AND total>0",(uid,))
    rows.sort(key=lambda r:r['correct']/max(r['total'],1)); return [r['topic'] for r in rows[:n]]

def due_topics(uid):
    now=datetime.datetime.utcnow().isoformat()
    return qa("SELECT topic FROM topic_mastery WHERE user_id=? AND next_review<=?",(uid,now))

def ups_topic(uid,topic,cor,tot):
    ex=q1("SELECT * FROM topic_mastery WHERE user_id=? AND topic=?",(uid,topic))
    now=datetime.datetime.utcnow().isoformat()
    if ex:
        iv=ex['interval_days']; iv=min(iv*2,30) if cor>0 else 1
        nr=(datetime.datetime.utcnow()+datetime.timedelta(days=iv)).isoformat()
        qx("UPDATE topic_mastery SET correct=correct+?,total=total+?,last_seen=?,next_review=?,interval_days=? WHERE user_id=? AND topic=?",
           (cor,tot,now,nr,iv,uid,topic))
    else:
        nr=(datetime.datetime.utcnow()+datetime.timedelta(days=1)).isoformat()
        qx("INSERT INTO topic_mastery(user_id,topic,correct,total,last_seen,next_review) VALUES(?,?,?,?,?,?)",
           (uid,topic,cor,tot,now,nr))

def fallback_qs(topic,n):
    b=[{'q':f'Core concept of "{topic}"?','opts':['A) Main framework','B) Minor detail','C) Unrelated idea','D) Deprecated'],'ans':'A','exp':'The core framework is essential.','topic':topic},
       {'q':f'Best way to study "{topic}"?','opts':['A) Re-reading','B) Active recall','C) Memorise only','D) Skip hard parts'],'ans':'B','exp':'Active recall builds stronger memory.','topic':topic},
       {'q':f'Handling difficult "{topic}" sections?','opts':['A) Skip','B) Break into chunks','C) Read faster','D) Summaries only'],'ans':'B','exp':'Chunking reduces cognitive load.','topic':topic}]
    return [dict(b[i%len(b)],q=f'Q{i+1}: {b[i%len(b)]["q"]}') for i in range(n)]

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/') 
def index(): return render_template('index.html')

@app.route('/static/<path:fn>')
def statics(fn): return send_from_directory(os.path.join(app.root_path,'static'),fn)

@app.route('/ping')
def ping(): return jsonify({'ok':True,'cohere':bool(COHERE_API_KEY),'ocr':bool(OCR_API_KEY)})

@app.route('/stats')
def stats():
    s=q1("SELECT * FROM system_stats WHERE id=1")
    return jsonify({'users':s['total_users'],'quizzes':s['total_quizzes'],'files':s['total_files']})

@app.route('/auth/check')
def auth_check():
    uid=session.get('user_id')
    if not uid: return jsonify({'authenticated':False})
    u=q1("SELECT * FROM users WHERE id=?",(uid,))
    if not u: session.clear(); return jsonify({'authenticated':False})
    upd_streak(uid,datetime.datetime.utcnow().isoformat())
    return jsonify({'authenticated':True,'user':u2d(q1("SELECT * FROM users WHERE id=?",(uid,)))})

@app.route('/auth/register',methods=['POST'])
def register():
    d=request.get_json() or {}
    name=(d.get('name') or '').strip(); uname=(d.get('username') or '').strip().lower()
    email=(d.get('email') or '').strip().lower(); pw=(d.get('password') or '')
    if not all([name,uname,email,pw]): return jsonify({'error':'All fields required'}),400
    if len(pw)<8: return jsonify({'error':'Password must be 8+ characters'}),400
    if q1("SELECT id FROM users WHERE email=?",(email,)): return jsonify({'error':'Email already registered'}),409
    if q1("SELECT id FROM users WHERE username=?",(uname,)): return jsonify({'error':'Username taken'}),409
    uid=str(uuid.uuid4()); now=datetime.datetime.utcnow().isoformat()
    qx("INSERT INTO users(id,name,username,email,password_hash,created_at,last_active) VALUES(?,?,?,?,?,?,?)",
       (uid,name,uname,email,hash_pw(pw),now,now))
    qx("UPDATE system_stats SET total_users=total_users+1 WHERE id=1")
    session.permanent=True; session['user_id']=uid
    return jsonify({'user':u2d(q1("SELECT * FROM users WHERE id=?",(uid,))),'message':'Account created'}),201

@app.route('/auth/login',methods=['POST'])
def login():
    d=request.get_json() or {}
    email=(d.get('email') or '').strip().lower(); pw=(d.get('password') or '')
    u=q1("SELECT * FROM users WHERE email=?",(email,))
    if not u or not chk_pw(pw,u['password_hash']): return jsonify({'error':'Invalid email or password'}),401
    upd_streak(u['id'],datetime.datetime.utcnow().isoformat())
    session.permanent=True; session['user_id']=u['id']
    return jsonify({'user':u2d(q1("SELECT * FROM users WHERE id=?",(u['id'],)))})

@app.route('/auth/logout',methods=['POST'])
def logout(): session.clear(); return jsonify({'message':'Logged out'})

@app.route('/user/me')
@require_login
def me(): return jsonify({'user':u2d(request.cu)})

@app.route('/user/xp',methods=['POST'])
@require_login
def add_xp():
    d=request.get_json() or {}; uid=request.cu['id']
    xp=int(d.get('amount',0)); dur=int(d.get('duration_min',0))
    qx("UPDATE users SET xp=xp+?,quizzes_done=quizzes_done+?,courses_done=courses_done+?,total_time_min=total_time_min+? WHERE id=?",
       (xp,int(d.get('quiz_done',0)),int(d.get('course_done',0)),dur,uid))
    u2=q1("SELECT xp FROM users WHERE id=?",(uid,)); rank,lv=rank_for(u2['xp'])
    qx("UPDATE users SET rank=?,level=? WHERE id=?",(rank,lv,uid))
    if xp>0:
        today=datetime.date.today().isoformat()
        ex=q1("SELECT * FROM study_sessions WHERE user_id=? AND date=?",(uid,today))
        if ex:
            topics=list(set(json.loads(ex['topics'] or '[]')+d.get('topics',[])))
            qx("UPDATE study_sessions SET duration=duration+?,xp_earned=xp_earned+?,topics=? WHERE user_id=? AND date=?",
               (dur,xp,json.dumps(topics),uid,today))
        else:
            qx("INSERT INTO study_sessions(user_id,date,duration,xp_earned,topics) VALUES(?,?,?,?,?)",
               (uid,today,dur,xp,json.dumps(d.get('topics',[]))))
    if d.get('quiz_done'): qx("UPDATE system_stats SET total_quizzes=total_quizzes+? WHERE id=1",(int(d.get('quiz_done',0)),))
    return jsonify({'user':u2d(q1("SELECT * FROM users WHERE id=?",(uid,)))})

@app.route('/user/topics',methods=['POST'])
@require_login
def upd_topics():
    d=request.get_json() or {}; uid=request.cu['id']
    for item in d.get('results',[]): ups_topic(uid,item['topic'],item['correct'],item['total'])
    return jsonify({'ok':True})

@app.route('/user/analytics')
@require_login
def analytics():
    uid=request.cu['id']
    topics=qa("SELECT topic,correct,total FROM topic_mastery WHERE user_id=? AND total>0 ORDER BY total DESC LIMIT 12",(uid,))
    thirty=(datetime.date.today()-datetime.timedelta(days=30)).isoformat()
    sess=qa("SELECT date,xp_earned,duration FROM study_sessions WHERE user_id=? AND date>=?",(uid,thirty))
    hm={s['date']:{'xp':s['xp_earned'],'min':s['duration']} for s in sess}
    wk=weak_topics(uid); dt=due_topics(uid)
    avg=round(sum(t['correct']/max(t['total'],1)*100 for t in topics)/max(len(topics),1)) if topics else 0
    return jsonify({'topics':[dict(t,mastery_pct=round(t['correct']/max(t['total'],1)*100)) for t in topics],
                    'heatmap':hm,'due_count':len(dt),'weak_topics':wk,
                    'total_time_min':request.cu['total_time_min'],'avg_score':avg})

@app.route('/leaderboard')
@require_login
def lb():
    uid=request.cu['id']; top=qa("SELECT * FROM users ORDER BY xp DESC LIMIT 50")
    total=q1("SELECT COUNT(*) n FROM users")['n']
    board=[{'rank':i+1,'name':u['name'],'username':u['username'],'xp':u['xp'],
            'rank_tier':rank_for(u['xp'])[0],'level':rank_for(u['xp'])[1],'is_me':u['id']==uid}
           for i,u in enumerate(top)]
    return jsonify({'total_users':total,'leaderboard':board})

@app.route('/quiz/saved')
@require_login
def saved_quizzes():
    uid=request.cu['id']
    qs=qa("SELECT id,title,questions,created_at FROM saved_quizzes WHERE user_id=? ORDER BY created_at DESC LIMIT 20",(uid,))
    return jsonify({'quizzes':[dict(q,questions=json.loads(q['questions'])) for q in qs]})

@app.route('/quiz/save',methods=['POST'])
@require_login
def save_quiz():
    d=request.get_json() or {}; qid=str(uuid.uuid4()); now=datetime.datetime.utcnow().isoformat()
    qx("INSERT INTO saved_quizzes(id,user_id,title,questions,created_at) VALUES(?,?,?,?,?)",
       (qid,request.cu['id'],d.get('title','Quiz'),json.dumps(d.get('questions',[])),now))
    return jsonify({'id':qid,'message':'Quiz saved'})

@app.route('/upload/extract',methods=['POST'])
@require_login
def upload_extract():
    files=request.files.getlist('files')
    if not files or not any(f.filename for f in files):
        return jsonify({'error':'No files received — please select at least one file'}),400
    combined=[]; filenames=[]; warnings=[]
    for f in files:
        if not f or not f.filename: continue
        try:
            text,err=proc_file(f)
            if err: warnings.append(f'{f.filename}: {err}')
            if text and text.strip():
                combined.append(f'=== {f.filename} ===\n{text.strip()}')
                filenames.append(f.filename)
        except Exception as e:
            warnings.append(f'{f.filename}: {e}')
    if not combined:
        return jsonify({'error':'No text extracted. '+(' | '.join(warnings) if warnings else 'Unknown error.')}),400
    qx("UPDATE system_stats SET total_files=total_files+? WHERE id=1",(len(filenames),))
    full='\n\n'.join(combined)
    return jsonify({'text':full,'char_count':len(full),'filenames':filenames,'warnings':warnings})

@app.route('/ai/generate-quiz',methods=['POST'])
@require_login
def gen_quiz():
    d=request.get_json() or {}
    text=(d.get('text') or '').strip(); title=(d.get('title') or 'Course').strip()
    nq=min(30,max(5,int(d.get('num_questions',10)))); diff=d.get('difficulty','intermediate')
    uid=request.cu['id']; wk=weak_topics(uid)
    wh=f'\nPrioritise weak areas: {", ".join(wk)}' if wk else ''
    prompt=f"""Expert AI curriculum engine. Generate exactly {nq} multiple-choice questions.
Difficulty:{diff} Topic:{title}{wh}
Material:{text[:4000] if text else f'Create questions about: {title}'}
Rules: each question needs a "topic" field (2-4 word label). 4 plausible options, one correct.
Output ONLY valid JSON no markdown:
{{"questions":[{{"q":"Question","opts":["A) o1","B) o2","C) o3","D) o4"],"ans":"A","exp":"Explanation.","topic":"Name"}}]}}"""
    try:
        raw=cohere(prompt); raw=re.sub(r'```(?:json)?\n?|\n?```','',raw).strip()
        m=re.search(r'\{.*\}',raw,re.DOTALL)
        if m: raw=m.group(0)
        qs=json.loads(raw).get('questions',[])
        if not qs: raise ValueError('empty')
        qx("UPDATE system_stats SET total_quizzes=total_quizzes+? WHERE id=1",(nq,))
        return jsonify({'questions':qs})
    except (json.JSONDecodeError,ValueError):
        return jsonify({'questions':fallback_qs(title,nq),'note':'fallback'})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/ai/adaptive-quiz',methods=['POST'])
@require_login
def adap_quiz():
    d=request.get_json() or {}; nq=min(20,max(5,int(d.get('num_questions',10)))); uid=request.cu['id']
    wk=weak_topics(uid,8); dt=[t['topic'] for t in due_topics(uid)]
    focus=list(set(dt+wk))
    if not focus: return jsonify({'error':'Complete some quizzes first to personalise!'}),400
    prompt=f"""Adaptive AI tutor. Generate exactly {nq} questions targeting: {', '.join(focus[:8])}
ONLY ask about listed topics. Start easier get harder. Each question needs "topic" field.
Output ONLY valid JSON: {{"questions":[{{"q":"Q","opts":["A) o1","B) o2","C) o3","D) o4"],"ans":"A","exp":"E.","topic":"T"}}]}}"""
    try:
        raw=cohere(prompt); raw=re.sub(r'```(?:json)?\n?|\n?```','',raw).strip()
        m=re.search(r'\{.*\}',raw,re.DOTALL)
        if m: raw=m.group(0)
        qs=json.loads(raw).get('questions',[])
        if not qs: raise ValueError('empty')
        return jsonify({'questions':qs,'focus_topics':focus})
    except Exception:
        return jsonify({'questions':fallback_qs(', '.join(focus[:2]),nq),'focus_topics':focus})

@app.route('/ai/explain-homework',methods=['POST'])
@require_login
def explain_hw():
    mode=request.form.get('mode','detailed'); desc=(request.form.get('description') or '').strip()
    ocr=''; ocr_err=None
    if 'image' in request.files:
        f=request.files['image']
        if f and f.filename: ocr,ocr_err=ocr_image(f,f.filename)
    combined='\n'.join(filter(None,[ocr,desc])).strip()
    if not combined:
        return jsonify({'error':f'Image unreadable ({ocr_err}). Type the problem instead.' if ocr_err
                        else 'Upload a photo OR type the problem in the text box.'}),400
    mmap={'simple':"simple — ELI5 style, use analogies",'detailed':'detailed with full steps',
          'exam':'exam-ready with formulas and common mistakes'}
    prompt=f"""Expert tutor. Explain in a {mmap.get(mode,'detailed')} way.
Problem: {combined}
Structure: 1.Problem Identification 2.Key Concepts 3.Step-by-Step Solution 4.Final Answer 5.Practice Problem"""
    try: return jsonify({'explanation':cohere(prompt),'ocr_text':ocr})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/ai/check-solution',methods=['POST'])
@require_login
def chk_sol():
    prob=(request.form.get('problem') or '').strip(); ans=(request.form.get('answer') or '').strip()
    ocr=''
    if 'image' in request.files:
        f=request.files['image']
        if f and f.filename: ocr,_=ocr_image(f,f.filename)
    work='\n'.join(filter(None,[ocr,ans])).strip()
    if not work: return jsonify({'error':'Write your answer or upload a photo of your working.'}),400
    prompt=f"""Strict but encouraging tutor reviewing student answer.
Problem:{prob or '(evaluate work on its own)'} Student work:{work}
1.CORRECT or INCORRECT (first sentence) 2.If wrong: specific mistake 3.Correct approach 4.Encouragement tip"""
    try:
        fb=cohere(prompt); ok=any(w in fb.lower()[:120] for w in ['correct','right','well done','excellent','perfect'])
        return jsonify({'feedback':fb,'is_correct':ok})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/ai/generate-summary',methods=['POST'])
@require_login
def gen_summary():
    d=request.get_json() or {}; text=(d.get('text') or '').strip(); title=(d.get('title') or 'Content').strip()
    prompt=f"""Expert study assistant. Create study package for: {title}
Content: {text[:5000] if text else f'Generate material about: {title}'}
Output ONLY valid JSON no markdown:
{{"summary":"3-5 paragraphs","key_points":["p1","p2"],"formulas":["f1"],"mind_map":{{"center":"Topic","branches":[{{"label":"B1","children":["s1","s2"]}}]}},"difficulty_areas":["hard concept"]}}"""
    try:
        raw=cohere(prompt,max_tok=3000); raw=re.sub(r'```(?:json)?\n?|\n?```','',raw).strip()
        m=re.search(r'\{.*\}',raw,re.DOTALL)
        if m: raw=m.group(0)
        return jsonify(json.loads(raw))
    except json.JSONDecodeError: return jsonify({'error':'AI response parse error — try again'}),500
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/ai/simulation',methods=['POST'])
@require_login
def sim():
    d=request.get_json() or {}; concept=(d.get('concept') or '').strip()
    if not concept: return jsonify({'error':'Enter a concept'}),400
    prompt=(f'Self-contained interactive HTML snippet (no external libs) demonstrating: "{concept}". '
            f'HTML+CSS+vanilla JS only. Interactive. Max 80 lines. Wrap in <div class="sim-container">. '
            f'Return ONLY the HTML snippet.')
    try:
        html=cohere(prompt,max_tok=2000)
        html=re.sub(r'<script[^>]*src=[^>]*>.*?</script>','',html,flags=re.DOTALL)
        return jsonify({'html':html})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/ai/mascot',methods=['POST'])
@require_login
def mascot():
    d=request.get_json(silent=True) or {}; page=d.get('page','dashboard'); ctx=d.get('context',{}); u=request.cu
    first=u['name'].split()[0]
    fb={'dashboard':f"Hey {first}! {u['xp']} XP so far — keep that streak alive!",'upload':'Drop your file and I will build a quiz in seconds!',
        'quiz':f"Think carefully before you tap, {first}!",'score':f"{'Brilliant!' if (ctx.get('score') or 0)>=70 else 'Keep pushing — every quiz sharpens you!'}",
        'homework':'Snap the problem and I will explain every step.','analytics':'Your weak spots today become your strengths tomorrow.',
        'leaderboard':f"Keep grinding! The top is waiting for you.",'settings':'Keep that streak alive. See you tomorrow!',
        'summary':'Your mind map is ready — every branch is a key concept.'}
    if (ctx.get('score') or ctx.get('weakTopic') or ctx.get('file_title')) and COHERE_API_KEY:
        parts=[f"user={first}",f"xp={u['xp']}",f"rank={u['rank']}"]
        if ctx.get('score') is not None: parts.append(f"just_scored={ctx['score']}%")
        if ctx.get('weakTopic'): parts.append(f"weak_at={ctx['weakTopic']}")
        if ctx.get('file_title'): parts.append(f"studying={ctx['file_title']}")
        try:
            msg=cohere(f"You are Brainy, cool mascot for Brainliant learning app. Context:{', '.join(parts)}. Page:{page}. Write ONE short encouraging sentence (max 18 words). No quotes.",max_tok=50).strip().strip('"\'')
            if msg and len(msg)>5: return jsonify({'message':msg[:160],'source':'ai'})
        except: pass
    return jsonify({'message':fb.get(page,f"Ready to learn something great, {first}!"),'source':'local'})

@app.errorhandler(413)
def too_large(e): return jsonify({'error':'File too large — max 50MB'}),413

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    print(f"\n{'='*50}\n  Brainliant — http://localhost:{port}\n  Cohere: {'OK' if COHERE_API_KEY else 'MISSING (add to .env)'}\n  OCR:    {'OK' if OCR_API_KEY else 'MISSING (add to .env)'}\n{'='*50}\n")
    app.run(debug=False,port=port,host='0.0.0.0')
