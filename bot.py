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
import uuid
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
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

## Safety
- Never ask for passwords, card numbers, or sensitive credentials through voice.
- Protect user privacy at all times.

## Functions — you have real tools; never pretend
- You can navigate the dashboard, look up crypto prices, check the user's balance/portfolio, analyze markets, pull transaction history, look up forex rates, and show alerts.
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
- Show an alert -> showAlert."""


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

ALL_TOOLS = [
    NAVIGATE_TOOL,
    SHOW_ALERT_TOOL,
    GET_CRYPTO_PRICE_TOOL,
    GET_PORTFOLIO_TOOL,
    GET_MARKET_ANALYSIS_TOOL,
    GET_TRANSACTIONS_TOOL,
    GET_FOREX_RATE_TOOL,
]


# =============================================================================
# LLM factory — Groq by default, OpenAI optional. Switchable via LLM_PROVIDER.
# =============================================================================

def _build_llm(max_tokens: int):
    """Build the conversational LLM service. Provider is env-switchable.

    LLM_PROVIDER=groq (default): GroqLLMService, model from GROQ_MODEL.
    LLM_PROVIDER=openai: OpenAILLMService, model from OPENAI_MODEL.
    Temperature is kept low (LLM_TEMPERATURE, default 0.4) for reliable tool calling.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))

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

    el_style = os.getenv("ELEVENLABS_STYLE", "0.45").strip()
    el_stability = os.getenv("ELEVENLABS_STABILITY", "0.35").strip()
    el_boost = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").strip()
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
        messages, tools=ToolsSchema(standard_tools=ALL_TOOLS)
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

    @llm.event_handler("on_function_calls_cancelled")
    async def on_calls_cancelled(_llm, calls):
        # User interrupted while a tool was in flight — drop the parked Futures so
        # nothing hangs.
        logger.info(f"tool calls cancelled: {[c.function_name for c in calls]}")
        for rid in list(pending_tools):
            fut = pending_tools.pop(rid, None)
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
        f"tools={[t.name for t in ALL_TOOLS]}"
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
