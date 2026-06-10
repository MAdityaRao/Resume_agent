import logging
import pathlib
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    JobProcess,
    AgentServer,
    Agent,
    AgentSession,
    TurnHandlingOptions,
    inference,
    cli,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("resume-agent")
logger.setLevel(logging.INFO)

RESUME_PATH = pathlib.Path(str(__file__)).resolve().parent / "resume.txt"
RESUME_TEXT = RESUME_PATH.read_text(encoding="utf-8").strip()

SYSTEM_PROMPT = f"""
You are a recruitment screening assistant. Your job is to evaluate whether a candidate
is a good fit for a job description.

You already have the candidate's resume:

--- CANDIDATE RESUME ---
{RESUME_TEXT}
--- END RESUME ---

WORKFLOW:
1. When the conversation starts, you will ask the user to paste the job description.
2. Once the user provides the job description, immediately evaluate the candidate.
3. Give a clear verdict: Fit, Partial Fit, or Not a Fit — with one sentence reason.
4. Then answer any follow-up questions the user has.

RULES:
- Keep every response short and direct — this is a voice conversation.
- Never make up experience that is not in the resume.
- If the user hasn't provided a JD yet, keep asking for it politely.
- Once JD is provided, do not ask for it again.
"""


class ResumeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_enter(self):
        await self.session.say(
            "Hi! Please paste the job description and I'll tell you if this candidate is a fit.",
            allow_interruptions=True,
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: VAD ready")


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
    )

    await session.start(agent=ResumeAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)