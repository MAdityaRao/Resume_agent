import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    JobContext,
    JobProcess,
    AgentServer,
    Agent,
    AgentSession,
    inference,
    cli,
)
from livekit.plugins import silero, sarvam
from persona import prompt

load_dotenv()
logger = logging.getLogger("resume-agent")
logger.setLevel(logging.INFO)


class ResumeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=prompt())


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

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=sarvam.TTS(
            api_key=os.getenv("SARVAM_API_KEY"),
            target_language_code="en-IN",
            model="bulbul:v3",
            speaker="rohan",
        ),
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.3,
        max_endpointing_delay=4.0,
    )

    await session.start(agent=ResumeAgent(), room=ctx.room)

    await session.say(
        "Hi, I'm Alex." " Ask me about Aditya. or paste a job description.",
        allow_interruptions=False,
    )


if __name__ == "__main__":
    cli.run_app(server)