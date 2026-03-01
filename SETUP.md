# Brainliant — Render Deployment Guide

## Why users kept getting logged out
The `SECRET_KEY` was set to `generateValue: true` in render.yaml.
This made Render create a **new secret key on every single deploy**.
Because Flask signs session cookies with the secret key, every deploy
instantly invalidated every user's session and remember-me cookie —
making it look like accounts had vanished.

**Fix:** `SECRET_KEY` must be a fixed value you set once and never change.

---

## One-time setup on Render (do this before deploying)

### Step 1 — Create a PostgreSQL database
1. Render Dashboard → **New → PostgreSQL**
2. Name: `brainliant-db`
3. Plan: Free
4. Click **Create Database**
5. Wait for it to say "Available"

> SQLite does NOT work on Render — its filesystem resets on every restart,
> wiping all user accounts. PostgreSQL persists forever.

### Step 2 — Deploy the web service
1. Render Dashboard → **New → Web Service**
2. Connect your GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`

### Step 3 — Set environment variables (CRITICAL)
Go to your web service → **Environment** tab and add:

| Variable | Value | Notes |
|----------|-------|-------|
| `SECRET_KEY` | `a-very-long-random-string-50-chars-min` | Generate at passwordsgenerator.net. **Set once, never change.** |
| `DATABASE_URL` | *(auto-filled)* | Link it: Environment → Add from database → brainliant-db |
| `COHERE_API_KEY` | your key | From cohere.com |
| `OCR_API_KEY` | your key | From ocr.space |
| `GOOGLE_CLIENT_ID` | your client ID | Optional. From console.cloud.google.com |
| `RENDER` | `true` | Enables HTTPS-only cookies |

### Step 4 — Link the database to the web service
1. Web service → Environment → **Add Environment Variable**
2. Key: `DATABASE_URL`
3. Value: click **"Link to database"** → select `brainliant-db` → **Connection String**

---

## Checklist before every deploy
- [ ] `SECRET_KEY` is set in Render environment (not auto-generated)
- [ ] `DATABASE_URL` points to the PostgreSQL database (not SQLite)
- [ ] `RENDER=true` is set (enables secure HTTPS cookies)

---

## Google Analytics
Your tag `G-H67NRCSEJP` is already in all three HTML pages (index, terms, privacy).
No action needed — it tracks automatically.
