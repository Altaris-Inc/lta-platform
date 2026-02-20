# Deployment Guide — Loan Tape Analyzer

## 1. Push to GitHub

### First time setup (new repo)
```bash
cd lta-platform
git init
git add .
git commit -m "Loan Tape Analyzer — full platform"

# Create repo on GitHub (github.com/new), then:
git remote add origin https://github.com/YOUR_ORG/loan-tape-analyzer.git
git branch -M main
git push -u origin main
```

### If repo already exists (Altaris-Inc/loan-tape-analyzer)
```bash
cd lta-platform
git add .
git commit -m "Update: custom strats, regression, templates, drill-down"
git push origin main
```

---

## 2. Deploy with Docker (Recommended)

### Option A: Any Linux Server (DigitalOcean, AWS EC2, Hetzner, etc.)

**Minimum specs:** 2 vCPU, 2GB RAM, 20GB disk (~$12/mo on DigitalOcean)

```bash
# 1. SSH into your server
ssh root@your-server-ip

# 2. Install Docker
curl -fsSL https://get.docker.com | sh

# 3. Clone your repo
git clone https://github.com/YOUR_ORG/loan-tape-analyzer.git
cd loan-tape-analyzer

# 4. Configure
cp .env.example .env
nano .env  # Change DB_PASSWORD!

# 5. Build & run
docker compose up -d --build

# 6. Check status
docker compose ps
docker compose logs -f frontend
```

**Your app is now live at:** `http://your-server-ip:8501`

### Option B: Add HTTPS with Caddy (production)

```bash
# Install Caddy reverse proxy
sudo apt install -y caddy

# Edit Caddy config
sudo nano /etc/caddy/Caddyfile
```

```
yourdomain.com {
    reverse_proxy localhost:8501
}

api.yourdomain.com {
    reverse_proxy localhost:8000
}
```

```bash
sudo systemctl restart caddy
```

Now accessible at `https://yourdomain.com` with auto SSL.

Update frontend env to point to public API:
```bash
# In .env, add:
API_URL=https://api.yourdomain.com
```

---

## 3. Deploy to Railway (Easiest — no server management)

[Railway](https://railway.app) runs Docker Compose natively.

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo
4. Railway auto-detects `docker-compose.yml`
5. Add environment variables in Railway dashboard:
   - `DB_PASSWORD` = strong password
   - `ANTHROPIC_API_KEY` = your key (optional)
6. Railway gives you a public URL automatically

**Cost:** ~$5-10/mo for light usage

---

## 4. Deploy to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Create 3 services:

**PostgreSQL Database:**
- Type: PostgreSQL
- Plan: Free tier or Starter ($7/mo)

**Backend (Web Service):**
- Environment: Docker
- Root Directory: `backend`
- Port: 8000
- Env vars: `DATABASE_URL` = Render provides this from the DB

**Frontend (Web Service):**
- Environment: Docker  
- Root Directory: `.` (root, since Dockerfile copies from parent)
- Dockerfile Path: `frontend/Dockerfile`
- Port: 8501
- Env vars: `API_URL` = your backend URL from Render

---

## 5. Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch backend
cd backend
fly launch --name lta-backend --region iad
fly secrets set DATABASE_URL="postgres://..."

# Launch frontend
cd ../frontend
fly launch --name lta-frontend --region iad
fly secrets set API_URL="https://lta-backend.fly.dev"
```

---

## 6. Local Development (Windows)

Already working! For reference:
```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2: Frontend  
cd frontend
pip install -r requirements.txt
streamlit run app_api.py
```

---

## Quick Reference

| Method | Cost | Difficulty | HTTPS | Custom Domain |
|--------|------|-----------|-------|---------------|
| DigitalOcean + Docker | ~$12/mo | Medium | With Caddy | Yes |
| Railway | ~$5-10/mo | Easy | Auto | Yes |
| Render | Free-$14/mo | Easy | Auto | Yes |
| Fly.io | ~$5-10/mo | Medium | Auto | Yes |
| AWS EC2 | ~$10-20/mo | Hard | Manual | Yes |

## Sharing with Others

Once deployed, share the URL. Users can:
1. Visit the URL → Register with name/email → Get API key
2. Login with API key
3. Upload their own CSV loan tapes
4. Each user's data is isolated (multi-tenant)

## Updating After Deployment

```bash
# On your server:
cd loan-tape-analyzer
git pull origin main
docker compose up -d --build
```

Or with Railway/Render: just push to GitHub — auto-deploys.
