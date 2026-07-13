import json
import logging
import os
import uuid
from typing import Any

import asyncpg
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from livekit.agents import (
    JobContext,
    JobProcess,
    AgentServer,
    Agent,
    AgentSession,
    RunContext,
    TurnHandlingOptions,
    function_tool,
    inference,
    cli,
)
from livekit.plugins import silero, sarvam
from persona import check_skill_match, prompt

load_dotenv()
logger = logging.getLogger("resume-agent")
logger.setLevel(logging.INFO)

# postgresql://user:password@host/dbname?sslmode=require
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
if not NEON_DATABASE_URL:
    logger.warning(
        "NEON_DATABASE_URL not found in environment — transcripts will fail to save. "
        "Check your .env file."
    )

_db_pool: asyncpg.Pool | None = None


async def get_db_pool() -> asyncpg.Pool:
    """Lazily create a single shared connection pool for the worker process."""
    global _db_pool
    if _db_pool is None:
        if not NEON_DATABASE_URL:
            raise RuntimeError(
                "NEON_DATABASE_URL is not set. Check that your .env file (in the same "
                "directory you run `uv run` from, or a parent of it) contains a line like:\n"
                "NEON_DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require\n"
                "Without this, asyncpg silently falls back to a local default connection "
                '(using your OS username as the db name — the "database \'adityarao\' does '
                'not exist" error is this exact symptom).'
            )
        _db_pool = await asyncpg.create_pool(
            NEON_DATABASE_URL,
            min_size=1,
            max_size=5,
            # Neon closes idle connections; keep pool resilient to that
            max_inactive_connection_lifetime=60,
        )
        logger.info("Neon DB pool created")
    return _db_pool


class TranscriptLog(BaseModel):
    """Validated shape of a row in resume_agent_logs before it's sent to Neon."""

    session_id: uuid.UUID
    visitor_name: str | None = None
    conversation: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("visitor_name")
    @classmethod
    def clean_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("conversation")
    @classmethod
    def non_empty_items_only(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Guard against garbage/mock objects ever reaching the DB layer
        for item in v:
            if not isinstance(item, dict):
                raise ValueError(f"conversation item must be a dict, got {type(item)}")
        return v


async def save_transcript(log: TranscriptLog) -> None:
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO resume_agent_logs (session_id, visitor_name, conversation)
                VALUES ($1, $2, $3::jsonb)
                """,
                log.session_id,
                log.visitor_name,
                json.dumps(log.conversation),
            )
        logger.info(f"Saved transcript for session {log.session_id} ({len(log.conversation)} items)")
    except Exception:
        logger.exception(f"Failed to save transcript for session {log.session_id}")


def extract_conversation_turns(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """session.history.to_dict()["items"] includes agent_handoff markers and
    agent_config_update entries (which dump the full system prompt). We only
    want the actual user/assistant messages in the stored transcript."""
    turns = []
    for item in raw_items:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        text = " ".join(content) if isinstance(content, list) else content
        turns.append(
            {
                "role": item.get("role"),
                "content": text,
                "interrupted": item.get("interrupted", False),
            }
        )
    return turns


class ResumeAgent(Agent):
    """Single agent for the whole conversation: collects the visitor's name
    first, then judges JDs / answers identity questions. No handoff — a
    two-agent handoff was triggering the framework's automatic post-tool
    LLM reply on the new agent (e.g. "Thank you, Swati!") on top of the
    greeting we already spoke manually, producing a contradictory double
    response."""

    def __init__(self, on_name_captured) -> None:
        self._on_name_captured = on_name_captured
        self._name_captured = False
        super().__init__(
            instructions=prompt(),
            tools=[check_skill_match],
        )

    async def on_enter(self) -> None:
        # The entrypoint already speaks the opening "what's your name?"
        # line manually — don't let the framework's default on_enter
        # trigger a second, unprompted LLM reply before the visitor has
        # said anything.
        pass

    @function_tool
    async def record_name(self, context: RunContext, name: str) -> str | None:
        """Call this as soon as the visitor states their name, before
        anything else — before judging a JD or answering a question, even
        if their first message also contains one of those.

        Args:
            name: The visitor's name, as they said it (e.g. "Rahul").
        """
        if self._name_captured:
            # Already recorded once this session — don't re-greet.
            return None

        candidate = name.strip().split()[0] if name.strip() else ""
        # Reject empty input or anything that's clearly not a name token
        # (too short, non-alphabetic, or an obvious non-answer).
        NON_NAME_WORDS = {"no", "none", "nothing", "why", "what", "who", "hi", "hello", "yes"}
        if (
            not candidate
            or not candidate.isalpha()
            or len(candidate) < 2
            or candidate.lower() in NON_NAME_WORDS
        ):
            return "That didn't sound like a name — ask the visitor for their name again."

        clean_name = candidate.capitalize()
        self._on_name_captured(clean_name)
        self._name_captured = True
        greeting = f"Hey {clean_name}! Ask me about Aditya, or paste a job description."
        # Speak the exact greeting ourselves so the LLM can't paraphrase a
        # different line, then return None so no automatic LLM reply is
        # generated on top of it (a non-empty/non-None return would still
        # get fed back to the LLM and produce a second, redundant reply).
        await context.session.say(greeting, allow_interruptions=False)
        return None


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: VAD ready")


server.setup_fnc = prewarm


# agent_name MUST match what route.js dispatches — this enables explicit dispatch on LiveKit Cloud
@server.rtc_session(agent_name="Aditya_Agent")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    await ctx.connect()

    session_id = uuid.uuid4()

    session = AgentSession(
        stt=sarvam.STT(
            api_key=os.getenv("SARVAM_API_KEY"),
            language="en-IN",
            model="saaras:v3",
            mode="transcribe",
        ),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=sarvam.TTS(
            api_key=os.getenv("SARVAM_API_KEY"),
            target_language_code="en-IN",
            model="bulbul:v3",
            speaker="priya",
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            endpointing={
                "mode": "dynamic",  # adapts to the visitor's actual pause patterns
                "min_delay": 0.3,   # was 0.3 — trims idle wait before the turn is considered done
                "max_delay": 3.0,   # was 4.0 — caps worst-case wait on long pauses
            },
           
        ),
    )

    captured_name: dict[str, str | None] = {"value": None}
    is_console = ctx.room.name == "console"

    def on_name_captured(name: str) -> None:
        captured_name["value"] = name

    async def write_transcript():
        raw_items = session.history.to_dict()["items"]
        conversation = extract_conversation_turns(raw_items)
        try:
            log = TranscriptLog(
                session_id=session_id,
                visitor_name=captured_name["value"],
                conversation=conversation,
            )
        except Exception:
            logger.exception(f"Transcript failed validation for session {session_id}")
            return
        await save_transcript(log)

    # Fires on graceful shutdown (participant disconnects, room closes, etc.)
    ctx.add_shutdown_callback(write_transcript)

    if is_console:
        # Local `console` testing — skip the name-collection step entirely.
        captured_name["value"] = "test"
        agent = ResumeAgent(on_name_captured)
        agent._name_captured = True  # suppress record_name's greeting; we say our own below
        await session.start(agent=agent, room=ctx.room)
        await session.say(
            "Hey test! Ask me about Aditya, or paste a job description.",
            allow_interruptions=False,
        )
    else:
        await session.start(agent=ResumeAgent(on_name_captured), room=ctx.room)
        await session.say(
            "Hi, I'm Priya, Aditya's assistant. Please tell me your name.",
            allow_interruptions=False,
        )


if __name__ == "__main__":
    cli.run_app(server)