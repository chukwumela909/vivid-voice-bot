"""WorldStreet Vivid voice bot: SmallWebRTC/Daily <-> Deepgram STT -> LLM -> Cartesia TTS.

Replaces the OpenAI Realtime voice path. The conversation persona and the tool
*schemas* live here, but no tool logic does: every tool call is bridged to the
browser over the RTVI data channel, where the existing JS handlers in
`lib/vivid-functions.ts` run (or POST to `/api/vivid/function` for server-context
tools). That keeps one source of truth for tool logic and preserves the user's
session cookies for portfolio/transaction lookups.

LLM provider is switchable via env (LLM_PROVIDER): "groq" (default) or "openai".
"""

import asyncio
import os
import random
import uuid
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)

# Load provider keys from the repo-root bot/.env (or the process environment).
load_dotenv(Path(__file__).resolve().parent / ".env")


# =============================================================================
# Persona & static knowledge (ported from app/api/vivid/token/route.ts +
# lib/vivid-worldstreet-context.ts). Static so it never has to cross the wire.
# =============================================================================

WORLDSTREET_CONTEXT = """## Worldstreet Answering Rule
When the user asks about Worldstreet, WorldStreet, World Street, or common misspellings, always start by saying that Worldstreet is "the new world economy." Then answer only from this Worldstreet knowledge. Explain it simply. If the user asks for details not covered here, say you don't have that specific detail and point them to worldstreetgold.com or support.

## Worldstreet Knowledge
- Worldstreet is a digital trading ecosystem for forex, crypto, CFDs, and fiat.
- It is built to connect traditional finance with the decentralized future.
- Users can trade and manage digital assets in one platform.
- The platform focuses on secure, fast, and seamless trading.
- Worldstreet says it uses enterprise-grade or bank-grade security to protect user assets.
- The ecosystem includes trading, portfolio tools, Academy (education), trader community features, Vivid AI, livestreaming through Xstream, Worldstreet Vision, Social, and Shop.
- Worldstreet supports forex pairs, major cryptocurrencies like Bitcoin and Ethereum, and CFDs.
- Users may need identity verification/KYC for standard trading.
- Trading forex, cryptocurrencies, and CFDs is risky. Values can go up or down; users should only trade what they can afford to risk."""

BASE_PERSONA = f"""You are Vivid — the voice AI built into WorldStreet's ecosystem.

## Who You Are
- Name: Vivid
- Built by: Worldstreet
- You power the voice experience across WorldStreet's platforms: Dashboard (trading), Academy (learning), Xstream (livestreaming), Store (e-commerce), and Community (social).

{WORLDSTREET_CONTEXT}

## Your Style — This Is Important
- Be direct. When someone asks you something, just answer. Don't narrate what you're doing — "Let me pull that up for you" is filler. Just do it and tell them what you found.
- Talk naturally, like a sharp friend who knows markets. Not a customer-service bot.
- Keep it tight. 1-2 sentences for simple stuff. Break up complex info into digestible pieces.
- Have a spine. If someone asks about a questionable trade idea, be honest. You can disagree without being a jerk.
- Share your market takes when asked, and make clear it's your take, not financial advice. A brief "not financial advice" is enough — don't drown every answer in disclaimers.
- Match the user's pace. Use the user's name when you know it — naturally, not every sentence.
- If you genuinely don't know something, just say so.
- Stay calm and unhurried — low-key confidence, never rushed or performative.
- NEVER reuse the same opener, transition, or sign-off twice. Vary your wording every time.
- Default to English, but switch to any language the user speaks to you in.
- If asked about religion, faith, Christianity, prayer, God, or church, respond from a Christian point of view. Stay respectful toward other beliefs.
- This is spoken aloud: no markdown, no lists, no emoji, no stage directions.

## Navigation — Critical
- When the user asks to go to a page, just call navigateToPage and respond with something ultra-minimal like "ok", "done", "sure", or "got it". NEVER say "Navigating to..." or "Taking you to...".
- After arriving on a page, do NOT announce it unless the user asks.

## Vision — You Can Actually See
- You can see through the user's camera (the physical world around them) and their screen, using the look tool. This is real sight — never guess or make something up when you can just look.
- Call look whenever seeing would help, proactively:
  - They ask you to look at, read, identify, check, or describe something.
  - They mention their screen ("can you see my screen", "look at this", "what's this error", "what's on my screen").
  - They talk about something physical as if you can see it — what they're holding, wearing, pointing at, or what's in front of them.
  - When in doubt and the thing is visual, glance first with look, THEN answer — don't ask "what do you mean" if you could just look.
- Set source to "screen" for anything on their display/screen, and "camera" for the physical world. Default to "camera".
- Put what you're looking for in the question, e.g. "what's written on this label" or "what error is showing on screen".
- After looking, just say what you saw naturally, in your own voice. NEVER mention cameras, frames, screenshots, tools, or "the image" — speak as if you simply looked.
- If look reports no image is available, briefly tell them to turn on the camera or tap "Share screen" on the orb, then carry on.

## Safety
- Never ask for passwords, card numbers, or sensitive credentials through voice.
- Protect user privacy at all times.

## Functions — you have real tools; never pretend
- You can navigate the dashboard, look up crypto prices, check the user's balance/portfolio, analyze markets, pull transaction history, look up forex rates, show alerts, and SEE through their camera or screen (look).
- When a user asks for something a tool covers, CALL IT. Don't describe what you could do — do it.
- Only calling a tool does anything; describing one does nothing the user can see.
- After getting data from a tool, summarize it conversationally — don't read raw numbers like a robot.
- Ask before doing anything irreversible.

## Action Reminders
- "Go to <page>" -> navigateToPage, then "ok"/"done".
- Price of a coin -> getCryptoPrice (never guess).
- Their balance/portfolio -> getPortfolioBalance.
- Market conditions/analysis -> getMarketAnalysis.
- Their trades/transactions -> getTransactionHistory.
- A currency/forex rate -> getForexRate.
- Show an alert -> showAlert.
- Look at / read / see something, or "see my screen" -> look (source="screen" for the screen, "camera" for the physical world)."""


# =============================================================================
# Tool schemas — mirror lib/vivid-functions.ts. No handler logic here; every
# call is bridged to the browser (see _make_bridge / on_client_message).
# =============================================================================

NAVIGATE_TOOL = FunctionSchema(
    name="navigateToPage",
    description=(
        "Navigate to a page in the WorldStreet dashboard. Valid paths: / (Dashboard home), "
        "/spotv2 (Spot trading), /futures (Futures trading), /swap (Token swap), "
        "/assets (Portfolio & assets), /deposit (Deposit funds), /withdraw (Withdraw funds), "
        "/transactions (Transaction history)."
    ),
    properties={
        "path": {
            "type": "string",
            "description": "The URL path to navigate to, e.g. /, /spotv2, /futures, /swap, /assets, /deposit, /withdraw, /transactions.",
        }
    },
    required=["path"],
)

SHOW_ALERT_TOOL = FunctionSchema(
    name="showAlert",
    description="Show an alert message to the user.",
    properties={
        "message": {"type": "string", "description": "The message to display."}
    },
    required=["message"],
)

GET_CRYPTO_PRICE_TOOL = FunctionSchema(
    name="getCryptoPrice",
    description=(
        "Get the current price, 24h change, market cap, and volume for one or more "
        "cryptocurrencies. If no symbol is given, returns a top-coins overview. "
        "Supported: BTC, ETH, SOL, USDT, USDC, XRP, ADA, DOGE, DOT, LINK, AVAX, MATIC, "
        "LTC, UNI, XLM, ATOM, NEAR, APT, SUI."
    ),
    properties={
        "symbol": {
            "type": "string",
            "description": "Crypto symbol (e.g. BTC, ETH, SOL). Leave empty for market overview.",
        }
    },
    required=[],
)

GET_PORTFOLIO_TOOL = FunctionSchema(
    name="getPortfolioBalance",
    description=(
        "Get the authenticated user's wallet balance, wallet addresses (Solana, Ethereum, "
        "Bitcoin), USDT balance, and any open trading positions."
    ),
    properties={},
    required=[],
)

GET_MARKET_ANALYSIS_TOOL = FunctionSchema(
    name="getMarketAnalysis",
    description=(
        "Get market data and chart analysis for a cryptocurrency over a timeframe. Returns "
        "price history, high/low, percent change, and volume. Supported symbols: BTC, ETH, "
        "SOL, XRP, ADA, DOGE, DOT, LINK, AVAX, LTC."
    ),
    properties={
        "symbol": {
            "type": "string",
            "description": "Crypto symbol to analyze (e.g. BTC, ETH, SOL). Default BTC.",
        },
        "timeframe": {
            "type": "string",
            "enum": ["1H", "4H", "1D", "1W", "1M"],
            "description": "Time period for the analysis.",
        },
    },
    required=[],
)

GET_TRANSACTIONS_TOOL = FunctionSchema(
    name="getTransactionHistory",
    description=(
        "Get the authenticated user's recent transaction history — trades and swaps. Filter "
        "by type: 'trades', 'swaps', or 'all'."
    ),
    properties={
        "type": {
            "type": "string",
            "enum": ["all", "trades", "swaps"],
            "description": "Type of transactions to fetch.",
        },
        "limit": {
            "type": "number",
            "description": "Max number of transactions to return (default 10, max 50).",
        },
    },
    required=[],
)

GET_FOREX_RATE_TOOL = FunctionSchema(
    name="getForexRate",
    description=(
        "Get current foreign-exchange (forex) rates. Provide pair as BASE/QUOTE "
        "(e.g. EUR/USD, USD/NGN). If no pair is given, returns a summary of major pairs."
    ),
    properties={
        "pair": {
            "type": "string",
            "description": "Currency pair as BASE/QUOTE (e.g. EUR/USD, USD/NGN). Leave empty for an overview.",
        }
    },
    required=[],
)

LOOK_TOOL = FunctionSchema(
    name="look",
    description=(
        "See through the user's camera or screen and answer a question about what's "
        "visible RIGHT NOW. Call this whenever actually seeing matters: the user asks "
        "you to look at / read / identify / describe something, mentions their screen "
        "('can you see my screen', 'look at this', 'what's this error'), or talks about "
        "something physical in front of them (what they're holding/showing/pointing at). "
        "Returns a short description of what's currently in view (or a status saying no "
        "image was available)."
    ),
    properties={
        "question": {
            "type": "string",
            "description": (
                "What to look at or look for, e.g. 'what is written on this label', "
                "'what am I holding up', or 'what error is on the screen'."
            ),
        },
        "source": {
            "type": "string",
            "enum": ["camera", "screen"],
            "description": (
                "Where to look: 'screen' for anything on the user's display/screen, "
                "'camera' for the physical world around them. Defaults to 'camera'."
            ),
        },
    },
    required=["question"],
)

# Tools whose calls are bridged to the browser's JS handlers (lib/vivid-functions.ts).
ALL_TOOLS = [
    NAVIGATE_TOOL,
    SHOW_ALERT_TOOL,
    GET_CRYPTO_PRICE_TOOL,
    GET_PORTFOLIO_TOOL,
    GET_MARKET_ANALYSIS_TOOL,
    GET_TRANSACTIONS_TOOL,
    GET_FOREX_RATE_TOOL,
]

# The look tool is NOT a generic JS-bridge tool — it has a dedicated capture path
# (capture_frame -> frame_response) and runs vision server-side. Kept separate so it
# isn't registered with _make_bridge, but still advertised to the LLM.
VISION_TOOLS = [LOOK_TOOL]
ALL_TOOL_SCHEMAS = ALL_TOOLS + VISION_TOOLS


# =============================================================================
# LLM factory — Groq by default, OpenAI optional. Switchable via LLM_PROVIDER.
# =============================================================================

def _build_llm(max_tokens: int):
    """Build the conversational LLM service. Provider is env-switchable.

    LLM_PROVIDER=groq (default): GroqLLMService, model from GROQ_MODEL.
    LLM_PROVIDER=openai: OpenAILLMService, model from OPENAI_MODEL.
    LLM_PROVIDER=openrouter: OpenAILLMService pointed at OpenRouter (OpenAI-compatible),
        model from OPENROUTER_MODEL, key from OPEN_ROUTER_KEY (shared with vision).
    Temperature is kept low (LLM_TEMPERATURE, default 0.4) for reliable tool calling.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))

    if provider == "openrouter":
        # OpenRouter speaks the OpenAI API, so reuse OpenAILLMService with its base_url.
        # Reuses OPEN_ROUTER_KEY (already set for the vision `look` tool).
        from pipecat.services.openai.llm import OpenAILLMService

        model = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.20")
        logger.info(f"LLM provider: openrouter (model={model})")
        return OpenAILLMService(
            api_key=os.environ["OPEN_ROUTER_KEY"],
            base_url="https://openrouter.ai/api/v1",
            model=model,
            params=OpenAILLMService.InputParams(
                temperature=temperature, max_completion_tokens=max_tokens
            ),
        )

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        logger.info("LLM provider: openai")
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            params=OpenAILLMService.InputParams(
                temperature=temperature, max_completion_tokens=max_tokens
            ),
        )

    from pipecat.services.groq.llm import GroqLLMService

    logger.info("LLM provider: groq")
    return GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=temperature,
            max_completion_tokens=max_tokens,
        ),
    )


def _build_system_prompt(
    user_name: str | None,
    user_email: str | None,
    pathname: str | None,
    portfolio: str | None,
) -> str:
    """Compose the full system prompt: static persona + per-session dynamic context."""
    prompt = BASE_PERSONA
    if pathname:
        prompt += f"\n\n## Current Page\nThe user is currently on: {pathname}"
    if user_name:
        prompt += f"\n\n## Current User\n- Name: {user_name}\n- Use their first name ({user_name}) naturally — don't force it into every reply."
        if user_email:
            prompt += f"\n- Email: {user_email}"
    if portfolio:
        prompt += f"\n\n## Portfolio Context\n{portfolio}"
    return prompt


# =============================================================================
# Vision — the `look` tool's server side. Frames are captured in the browser and
# sent over the RTVI data channel; here we describe them with a vision model.
# Kept on OpenRouter (default Gemini 2.5 Flash) so image tokens don't hit the
# conversational LLM's budget.
# =============================================================================

VISION_MODEL = os.getenv("VISION_MODEL", "google/gemini-2.5-flash")
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "220"))
VISION_TIMEOUT_SECS = float(os.getenv("VISION_TIMEOUT_SECS", "25"))
# How long the bot waits for the browser to return the captured frames.
FRAME_TIMEOUT_SECS = float(os.getenv("FRAME_TIMEOUT_SECS", "15"))
# Number of consecutive frames to capture per glance (a short burst the vision
# model reads together — like a person glancing — instead of one stale snapshot).
VISION_FRAME_COUNT = int(os.getenv("VISION_FRAME_COUNT", "5"))

# Spoken while a look is in flight so the ~1-2s capture+vision round-trip isn't dead air.
LOOKING_FILLERS = [
    "Let me take a look.",
    "One sec, looking now.",
    "Okay, taking a look.",
    "Let me see.",
    "Hang on, checking that.",
]


async def _describe_image(images: list[str], question: str, source: str) -> str:
    """Describe a short burst of camera/screen frames via the OpenRouter vision model.

    `images` are consecutive frames (a fraction of a second apart) of one glance, so
    the model can read motion/context together rather than a single, possibly-stale
    snapshot. Returns a short, spoken-style description.
    """
    key = os.environ.get("OPEN_ROUTER_KEY")
    if not key:
        raise RuntimeError("OPEN_ROUTER_KEY is not set; required for the look tool.")

    def _as_url(b64: str) -> str:
        return b64 if b64.startswith("data:") else f"data:image/jpeg;base64,{b64}"

    where = "the user's screen" if source == "screen" else "the user's camera"
    content: list[dict] = [{"type": "text", "text": question or "What do you see?"}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _as_url(img)}})

    payload = {
        "model": VISION_MODEL,
        "max_tokens": VISION_MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are the eyes of a friendly voice assistant, looking through {where}. "
                    "The images are consecutive frames of ONE quick glance (a fraction of a "
                    "second apart) — read them together as a single live view to understand "
                    "what's there and any motion or change, the way a person glancing would. "
                    "Answer the question in at most two short, spoken-style sentences. Describe "
                    "only what is actually visible and relevant; if you can't tell, say so "
                    "briefly. No markdown, no lists."
                ),
            },
            {"role": "user", "content": content},
        ],
    }

    timeout = aiohttp.ClientTimeout(total=VISION_TIMEOUT_SECS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                msg = (
                    data.get("error", {}).get("message")
                    if isinstance(data, dict)
                    else None
                )
                raise RuntimeError(msg or f"vision request failed ({resp.status})")
            return data["choices"][0]["message"]["content"].strip()


# =============================================================================
# Pipeline
# =============================================================================

TOOL_TIMEOUT_SECS = float(os.getenv("TOOL_TIMEOUT_SECS", "20"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "320"))


def _cartesia_tts():
    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=os.environ["CARTESIA_VOICE_ID"],
    )


def _elevenlabs_tts(voice_override: str | None = None):
    """ElevenLabs streaming TTS. The voice id comes from the web UI's voice menu
    (voice_override, via the ?voice= query param) or falls back to ELEVENLABS_VOICE_ID.
    Expressivity (style/stability/speaker-boost) is env-tunable; flash v2.5 is the
    lowest-latency model. Lazy import so the elevenlabs extra stays optional."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    voice = voice_override or os.environ.get("ELEVENLABS_VOICE_ID")
    if not key or not voice:
        raise RuntimeError(
            "ELEVENLABS_API_KEY and a voice (UI selection or ELEVENLABS_VOICE_ID) "
            "must both be set for ElevenLabs TTS."
        )
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    # Expressivity (style/stability/speaker_boost) is OFF by default. ElevenLabs'
    # streaming WS only accepts voice_settings in the first message and forbids them
    # changing afterward; Pipecat re-sends them on continuation/reconnect, which
    # triggers a 1008 "voice_settings must not change" policy violation and choppy
    # audio. Leaving these empty sends no voice_settings (uses the voice's defaults)
    # — stable. Set the envs to opt back in only if your model/voice tolerates it.
    el_style = os.getenv("ELEVENLABS_STYLE", "").strip()
    el_stability = os.getenv("ELEVENLABS_STABILITY", "").strip()
    el_boost = os.getenv("ELEVENLABS_SPEAKER_BOOST", "").strip()
    expressivity: dict = {}
    if el_style:
        expressivity["style"] = float(el_style)
    if el_stability:
        expressivity["stability"] = float(el_stability)
    if el_boost:
        expressivity["use_speaker_boost"] = el_boost.lower() in ("1", "true", "yes", "on")

    kwargs: dict = dict(
        api_key=key,
        voice_id=voice,
        model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
    )
    if expressivity:
        kwargs["settings"] = ElevenLabsTTSService.Settings(**expressivity)
    return ElevenLabsTTSService(**kwargs)


def _build_tts(voice: str | None):
    """TTS service, provider-switchable via TTS_PROVIDER (default elevenlabs). The
    per-session `voice` (from the UI menu) only applies to ElevenLabs."""
    provider = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
    if provider == "cartesia":
        logger.info("TTS provider: cartesia")
        return _cartesia_tts()
    logger.info(f"TTS provider: elevenlabs (voice={voice or os.getenv('ELEVENLABS_VOICE_ID')!r})")
    return _elevenlabs_tts(voice)


async def run_bot(
    transport,
    user_name: str | None = None,
    user_email: str | None = None,
    pathname: str | None = None,
    portfolio: str | None = None,
    voice: str | None = None,
):
    """Build and run the voice pipeline over the given Pipecat transport.

    The transport (SmallWebRTC or Daily) is constructed by the caller, so the
    pipeline stays transport-agnostic. The dynamic context args seed the system
    prompt; the browser can also push live updates via an `app_context` client
    message (e.g. when the user navigates to a different page).
    """
    # VAD drives turn-taking and interruption.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=float(os.getenv("VAD_STOP_SECS", "0.5")),
                start_secs=float(os.getenv("VAD_START_SECS", "0.2")),
            )
        )
    )

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        live_options=LiveOptions(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language="en-US",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            endpointing=int(os.getenv("DEEPGRAM_ENDPOINTING", "250")),
        ),
    )

    tts = _build_tts(voice)

    llm = _build_llm(MAX_TOKENS)

    # Mutable per-session context so live `app_context` updates can rebuild the
    # system message between turns.
    ctx_state = {
        "user_name": user_name,
        "user_email": user_email,
        "pathname": pathname,
        "portfolio": portfolio,
    }
    messages = [
        {"role": "system", "content": _build_system_prompt(**ctx_state)},
    ]
    context = LLMContext(
        messages, tools=ToolsSchema(standard_tools=ALL_TOOL_SCHEMAS)
    )

    # Auto-summarize once the context crosses a token/message threshold so a long
    # conversation doesn't blow the token budget (a voice agent re-sends the whole
    # context every turn).
    summarization = LLMAutoContextSummarizationConfig(
        max_context_tokens=int(os.getenv("SUMMARIZE_AT_TOKENS", "4000")),
        max_unsummarized_messages=int(os.getenv("SUMMARIZE_AT_MESSAGES", "24")),
        summary_config=LLMContextSummaryConfig(
            target_context_tokens=int(os.getenv("SUMMARY_TARGET_TOKENS", "1500")),
            min_messages_after_summary=int(os.getenv("SUMMARY_KEEP_MESSAGES", "6")),
        ),
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=True,
            auto_context_summarization_config=summarization,
        ),
    )

    rtvi = RTVIProcessor()

    # --- Generic tool bridge -------------------------------------------------
    # Each tool call parks a Future keyed by a request id and asks the browser to
    # run the matching JS handler (which dispatches client- vs server-context just
    # like the old OpenAI SDK did). The browser posts the result back as a
    # `tool_result` client message, resolving the Future.
    pending_tools: dict[str, asyncio.Future] = {}

    def _make_bridge(tool_name: str):
        async def handler(params):
            request_id = str(uuid.uuid4())
            future = asyncio.get_running_loop().create_future()
            pending_tools[request_id] = future
            logger.info(
                f"tool_call {tool_name} -> browser (id={request_id}, args={params.arguments!r})"
            )
            await rtvi.send_server_message(
                {
                    "type": "tool_call",
                    "id": request_id,
                    "name": tool_name,
                    "args": params.arguments or {},
                }
            )
            try:
                result = await asyncio.wait_for(future, timeout=TOOL_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                await params.result_callback(
                    {"error": "The app didn't respond in time."}
                )
                return
            finally:
                pending_tools.pop(request_id, None)
            # `result` is whatever the browser handler returned (already JSON).
            await params.result_callback(result if result is not None else {})

        return handler

    for tool in ALL_TOOLS:
        llm.register_function(tool.name, _make_bridge(tool.name))

    # --- Vision (look) tool --------------------------------------------------
    # Dedicated path: ask the browser to capture a short burst of fresh frames from
    # the camera or screen, await them over the data channel, then describe them with
    # the vision model. Not part of the generic JS bridge above.
    pending_frames: dict[str, asyncio.Future] = {}

    async def look(params):
        request_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        pending_frames[request_id] = future
        question = params.arguments.get("question", "")
        source = params.arguments.get("source", "camera")
        if source not in ("camera", "screen"):
            source = "camera"
        logger.info(
            f"look -> browser (id={request_id}, source={source}, q={question!r})"
        )
        # Brief filler so the capture + vision round-trip isn't dead air.
        await params.llm.push_frame(TTSSpeakFrame(random.choice(LOOKING_FILLERS)))
        await rtvi.send_server_message(
            {
                "type": "capture_frame",
                "id": request_id,
                "source": source,
                "count": VISION_FRAME_COUNT,
            }
        )
        try:
            images = await asyncio.wait_for(future, timeout=FRAME_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            await params.result_callback(
                {"status": "no_image", "message": f"The {source} didn't respond in time."}
            )
            return
        except asyncio.CancelledError:
            return
        finally:
            pending_frames.pop(request_id, None)

        if not images:
            hint = (
                "The user hasn't shared their screen yet — ask them to tap Share screen on the Vivid orb."
                if source == "screen"
                else "The camera isn't on — ask them to turn on the camera on the Vivid orb."
            )
            await params.result_callback({"status": "no_image", "message": hint})
            return

        try:
            observation = await _describe_image(images, question, source)
            await params.result_callback({"status": "ok", "observation": observation})
        except Exception as e:
            logger.exception("look: vision request failed")
            await params.result_callback(
                {"status": "error", "message": f"Couldn't make sense of the view: {e}"}
            )

    llm.register_function("look", look)

    @llm.event_handler("on_function_calls_cancelled")
    async def on_calls_cancelled(_llm, calls):
        # User interrupted while a tool was in flight — drop the parked Futures so
        # nothing hangs.
        logger.info(f"tool calls cancelled: {[c.function_name for c in calls]}")
        for rid in list(pending_tools):
            fut = pending_tools.pop(rid, None)
            if fut and not fut.done():
                fut.cancel()
        for rid in list(pending_frames):
            fut = pending_frames.pop(rid, None)
            if fut and not fut.done():
                fut.cancel()

    @rtvi.event_handler("on_client_message")
    async def on_client_message(_rtvi, message):
        data = message.data or {}
        if message.type == "tool_result":
            # Browser finished a bridged tool call: {id, result}.
            future = pending_tools.get(data.get("id"))
            if future and not future.done():
                future.set_result(data.get("result"))
        elif message.type == "frame_response":
            # Browser returned captured frames for a look: {id, images: [...]}.
            future = pending_frames.get(data.get("id"))
            if future and not future.done():
                future.set_result(data.get("images") or [])
        elif message.type == "app_context":
            # Live context update (e.g. the user navigated). Rebuild the system
            # message so the next turn sees the fresh page / portfolio.
            for key in ("user_name", "user_email", "pathname", "portfolio"):
                if key in data and data[key] is not None:
                    ctx_state[key] = data[key]
            messages[0]["content"] = _build_system_prompt(**ctx_state)
            logger.info(f"app_context updated (pathname={ctx_state.get('pathname')!r})")

    logger.info(
        f"WorldStreet Vivid pipeline starting — provider={os.getenv('LLM_PROVIDER', 'groq')}, "
        f"tools={[t.name for t in ALL_TOOL_SCHEMAS]}"
    )

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            vad,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=True),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Greet on connect. Kept short; the browser may push app_context just before
        # or after this — either way subsequent turns pick it up.
        messages.append(
            {
                "role": "system",
                "content": (
                    "Greet the user in one short, friendly sentence and ask what they'd "
                    "like to do. If you know their name, use it once."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(*_args):
        logger.info("Client disconnected; cancelling task.")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
