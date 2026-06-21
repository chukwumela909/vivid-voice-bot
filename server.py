"""FastAPI signaling server for the WorldStreet Vivid voice bot.

Two transports, switched by the TRANSPORT env var:
  - "smallwebrtc" (default): the browser's Pipecat SmallWebRTC transport speaks
    POST/PATCH against /api/offer (SDP + ICE trickle). Peer-to-peer; good for local
    dev. Needs STUN/TURN to traverse NAT — see DEPLOY notes.
  - "daily": the browser calls POST /connect to get a Daily room + token. Managed
    media infra (SFU + TURN); recommended for cloud.

Provider API keys never leave this process. One bot pipeline is spawned per peer
connection. The browser passes per-session context (user name/email, current page,
a short portfolio summary) as query params; they seed the bot's system prompt.
"""

import json
import os

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from bot import run_bot

TRANSPORT = os.getenv("TRANSPORT", "smallwebrtc").lower()

# Restrict to the web app origin(s) in production. Comma-separated list, or "*"
# for local dev. e.g. ALLOWED_ORIGINS="https://app.worldstreet.com".
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
] or ["*"]


def _load_ice_servers() -> list[IceServer]:
    """Build the ICE server list from WEBRTC_ICE_SERVERS (JSON array of url strings
    or {urls, username, credential} objects). Empty/unset -> host candidates only."""
    raw = os.getenv("WEBRTC_ICE_SERVERS", "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"WEBRTC_ICE_SERVERS is not valid JSON ({e}); ignoring.")
        return []

    servers: list[IceServer] = []
    for entry in entries:
        if isinstance(entry, str):
            servers.append(IceServer(urls=entry))
        elif isinstance(entry, dict) and entry.get("urls"):
            servers.append(
                IceServer(
                    urls=entry["urls"],
                    username=entry.get("username"),
                    credential=entry.get("credential"),
                )
            )
        else:
            logger.warning(f"Skipping malformed ICE server entry: {entry!r}")
    logger.info(f"Loaded {len(servers)} ICE server(s).")
    return servers


ICE_SERVERS = _load_ice_servers()
webrtc = SmallWebRTCRequestHandler(ice_servers=ICE_SERVERS)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "transport": TRANSPORT}


@app.post("/api/offer")
async def offer(
    request: dict,
    background_tasks: BackgroundTasks,
    user_name: str = "",
    user_email: str = "",
    pathname: str = "",
    portfolio: str = "",
):
    # The browser appends ?user_name=...&pathname=... so the bot can personalize the
    # first greeting. Spawn the pipeline only for a brand-new connection.
    async def on_new_connection(connection):
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )
        background_tasks.add_task(
            run_bot,
            transport,
            user_name=user_name or None,
            user_email=user_email or None,
            pathname=pathname or None,
            portfolio=portfolio or None,
        )

    return await webrtc.handle_web_request(
        SmallWebRTCRequest.from_dict(request), on_new_connection
    )


@app.patch("/api/offer")
async def offer_patch(request: dict):
    candidates = [IceCandidate(**c) for c in request.get("candidates", [])]
    await webrtc.handle_patch_request(
        SmallWebRTCPatchRequest(pc_id=request["pc_id"], candidates=candidates)
    )
    return {"status": "ok"}


@app.post("/connect")
async def connect(
    background_tasks: BackgroundTasks,
    user_name: str = "",
    user_email: str = "",
    pathname: str = "",
    portfolio: str = "",
):
    """Daily transport: create a room + tokens, spawn the bot, return client creds.

    Daily imports are deferred so SmallWebRTC-only setups (e.g. local Windows, where
    daily-python has no wheels) never need the Daily dependency installed.
    """
    import time

    import aiohttp
    from pipecat.transports.daily.transport import DailyParams, DailyTransport
    from pipecat.transports.daily.utils import (
        DailyRESTHelper,
        DailyRoomParams,
        DailyRoomProperties,
    )

    api_key = os.environ["DAILY_API_KEY"]
    async with aiohttp.ClientSession() as session:
        helper = DailyRESTHelper(daily_api_key=api_key, aiohttp_session=session)
        room = await helper.create_room(
            DailyRoomParams(
                properties=DailyRoomProperties(
                    exp=time.time() + 3600, eject_at_room_exp=True
                )
            )
        )
        bot_token = await helper.get_token(room.url, 3600)
        client_token = await helper.get_token(room.url, 3600)

    transport = DailyTransport(
        room.url,
        bot_token,
        "Vivid",
        DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    background_tasks.add_task(
        run_bot,
        transport,
        user_name=user_name or None,
        user_email=user_email or None,
        pathname=pathname or None,
        portfolio=portfolio or None,
    )
    return {"room_url": room.url, "token": client_token}


if __name__ == "__main__":
    port = int(os.getenv("BOT_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
