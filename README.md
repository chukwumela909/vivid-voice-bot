# WorldStreet Vivid voice bot

Standalone Pipecat voice agent that powers the Vivid orb in the WorldStreet web app
(replacing the OpenAI Realtime path). It runs the **Deepgram STT → LLM (Groq/OpenAI)
→ Cartesia TTS** loop and bridges every tool call to the browser over the RTVI data
channel.

This is its **own repo / service** — it has no code dependency on the web app. The
only link is the browser, which connects to this bot's WebRTC endpoint and runs the
tool handlers. Deploy it independently (see `DEPLOY.md`).

## How tool calling works (the important part)

This bot defines the tool **schemas** (`bot.py` → `ALL_TOOLS`) but contains **no tool
logic**. When the LLM calls a tool, the bot sends a `tool_call` message to the browser
and waits. In the **web app repo**, `components/vivid/pipecat-provider.tsx` runs the
matching handler from `lib/vivid-functions.ts` — dispatching client- vs server-context
exactly like the old OpenAI SDK did (`executionContext: "server"` →
`POST /api/vivid/function`, otherwise the in-browser JS handler) — and posts the result
back as `tool_result`.

This keeps **one source of truth** for tool logic in the web app and preserves the
user's session cookies for portfolio/transaction lookups (those run in the browser).

To add a tool: add a `FunctionSchema` to `ALL_TOOLS` here, and make sure the matching
`createVividFunction` exists in the web app's `lib/vivid-functions.ts`. No handler code
lives in this repo.

## Run locally (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env   # then fill in DEEPGRAM / GROQ / CARTESIA keys
.\.venv\Scripts\python.exe server.py   # http://localhost:7860, GET /health
```

Then run the web app with `NEXT_PUBLIC_BOT_OFFER_URL=http://localhost:7860/api/offer`
and click the Vivid orb.

## Switching the LLM

```
LLM_PROVIDER=groq    GROQ_MODEL=openai/gpt-oss-20b     # default
LLM_PROVIDER=openai  OPENAI_MODEL=gpt-4o               # stronger tool-calling
```

## Deployment

See `DEPLOY.md` for the full Coolify guide. In short: deploy this repo as a Dockerfile
app, use `TRANSPORT=daily` (no NAT/TURN setup), give it an HTTPS domain, set
`ALLOWED_ORIGINS` to the web app's origin, and point the web app's
`NEXT_PUBLIC_BOT_OFFER_URL` at this bot's `/api/offer`.
