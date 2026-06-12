import logging

from dotenv import load_dotenv
import os
from livekit.agents import (
    JobContext,
    JobProcess,
    AgentServer,
    Agent,
    AgentSession,
    inference,
    cli,
)
from livekit.plugins import silero,sarvam
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from persona import prompt
load_dotenv()
logger = logging.getLogger("resume-agent")
logger.setLevel(logging.INFO)
class ResumeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=prompt())

    async def on_enter(self):
        await self.session.say(
            """
            I'm sharanya."  " aditya's personal assitant.
            ask me about him or give a JD.
            """,
            allow_interruptions=False,
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
        stt=inference.STT(model="cartesia/ink-whisper"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=sarvam.TTS(
            api_key=os.getenv("SARVAM_API_KEY"),
            target_language_code="en-IN",
            model="bulbul:v3",
            speaker="priya",
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(agent=ResumeAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)