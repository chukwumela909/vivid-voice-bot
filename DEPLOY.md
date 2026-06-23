# Deploying the WorldStreet Vivid voice bot on Coolify

This repo is the **voice bot only** — a standalone Python/FastAPI service. It deploys
independently of the WorldStreet web app and shares no code with it. The only link is
the browser, which connects to this bot's WebRTC endpoint.

| Resource | Repo | Build pack | Base Directory | Port |
|---|---|---|---|---|
| **bot** (this) | this repo | Dockerfile | `/` | 7860 |
| **web** (WorldStreet) | the app repo | Nixpacks | `/` | 3000 |

The web app's `NEXT_PUBLIC_BOT_OFFER_URL` points at this bot's public domain. Nothing
in the web repo changes except those env vars (step 5).

**Use the `daily` transport.** It hands the media (audio over UDP) to Daily's infra
(SFU + TURN), so you skip all the NAT/ICE/coturn pain SmallWebRTC needs inside a
bridged Docker container. Get a free `DAILY_API_KEY` from daily.co.

---

## 1. Create the repo and push

```bash
cd vivid-voice-bot
git init
git add .
git commit -m "WorldStreet Vivid voice bot"
gh repo create vivid-voice-bot --private --source=. --remote=origin --push
```

`.env` is gitignored, so no secrets get committed.

## 2. Add it as a Coolify resource

Coolify → your project → **+ New Resource → Application → Dockerfile**.

- **Source:** this repo.
- **Base Directory:** `/`  (the Dockerfile is at the repo root).
- **Port:** `7860`.

*(Alternative — Docker Compose: add a "Docker Compose" resource with path
`docker-compose.yml`. Only needed if you want SmallWebRTC + a self-hosted coturn TURN
relay. With Daily you don't.)*

## 3. Environment variables (Coolify → this service → Environment)

Copy from `.env.example`. Minimum for Daily:

| Var | Value |
|---|---|
| `DEEPGRAM_API_KEY` | your Deepgram key |
| `CARTESIA_API_KEY` | your Cartesia key |
| `CARTESIA_VOICE_ID` | a Cartesia voice id |
| `GROQ_API_KEY` | your Groq key (default LLM) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` (or a larger model) |
| `LLM_PROVIDER` | `groq` (or `openai` + `OPENAI_API_KEY`/`OPENAI_MODEL`, or `openrouter` + `OPEN_ROUTER_KEY`/`OPENROUTER_MODEL`) |
| `TRANSPORT` | `daily` |
| `DAILY_API_KEY` | your Daily key |
| `ALLOWED_ORIGINS` | `https://<your-web-domain>` (lock CORS; `*` to start) |

## 4. Domain & HTTPS (mandatory)

Assign a domain in Coolify (e.g. `https://vivid-bot.yourdomain.com`); it provisions
Let's Encrypt TLS automatically. Browsers block the mic + WebRTC on plain HTTP.

Verify: `curl https://vivid-bot.yourdomain.com/health` →
`{"status":"ok","transport":"daily"}`.

## 5. Point the web app at the bot

On the **WorldStreet web resource** in Coolify, add these env vars and **trigger a
full rebuild** (the web builds with Nixpacks, which bakes `NEXT_PUBLIC_*` at build time
— a restart alone won't pick them up):

```
NEXT_PUBLIC_BOT_OFFER_URL=https://vivid-bot.yourdomain.com/api/offer
NEXT_PUBLIC_TRANSPORT=daily          # must match this bot's TRANSPORT
NEXT_PUBLIC_WEBRTC_ICE_SERVERS=["stun:stun.l.google.com:19302"]   # ignored when daily
```

If your Coolify shows an "Available at Buildtime / Build Variable" toggle per var,
enable it for these three.

## 6. Verify end-to-end

1. Open the web app, sign in, click the Vivid orb → connecting → ready, it greets you.
2. Speak → you hear a reply; the orb animates to the voice.
3. "What's the price of BTC?" and "go to futures" exercise the tool bridge.
4. "What's my portfolio?" proves the in-browser session-cookie auth still works.

## 7. Auto-deploy

Enable Coolify's GitHub webhook so pushes to this repo's `main` redeploy the bot. The
web app only needs a rebuild if you change the bot's domain.

---

## If you must use SmallWebRTC instead of Daily

Inside a bridged Docker container, STUN alone is unreliable — you need a TURN relay.
Uncomment the `coturn` service in `docker-compose.yml`, open **UDP/TCP 3478** and
**UDP 49160–49200** on the server firewall, set `--external-ip` to the server's public
IP, then set on **both** bot and web:

```json
WEBRTC_ICE_SERVERS / NEXT_PUBLIC_WEBRTC_ICE_SERVERS =
[
  "stun:stun.l.google.com:19302",
  {"urls":"turn:turn.yourdomain.com:3478","username":"USER","credential":"PASS"}
]
```

and `TRANSPORT=smallwebrtc` / `NEXT_PUBLIC_TRANSPORT=smallwebrtc`. Daily avoids all of
this — prefer it unless you have a reason not to.
