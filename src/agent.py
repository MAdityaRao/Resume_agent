import logging
import os
from dotenv import load_dotenv
from livekit import rtc
import json
import asyncio
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    utils,
)
from livekit.plugins import noise_cancellation, silero, elevenlabs

logger = logging.getLogger("agent")
load_dotenv(".env.local")

RESUME_FILE_PATH = "resume.txt"
RESUME_CONTENT = ""

try:
    if os.path.exists(RESUME_FILE_PATH):
        with open(RESUME_FILE_PATH, "r", encoding="utf-8") as f:
            RESUME_CONTENT = f.read()
        logger.info("resume.txt loaded successfully.")
    else:
        logger.warning("resume.txt not found! Using placeholder text.")
        RESUME_CONTENT = "No resume text found. Please ensure resume.txt exists."
except Exception as e:
    logger.error(f"Failed to read resume.txt: {e}")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""
You are Aditya AI Assistant, acting as a candidate in an interview.
My resume is: {RESUME_CONTENT}
Your task is to answer interview questions based strictly on my resume.
Rules:
- Keep each response under 30-45 words.
- Tone: professional, concise, business-like.
- Do not add explanations or extra suggestions.
- dont answer unnecessary questions, only answer based on the resume.
- Answer directly and naturally as if you are the candidate in a live interview.
- if jd is one line like "i want a python developer", then say how you are a good fit based on your resume, but keep it concise.
"""
 )
server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    await ctx.connect()
    
    logger.info(f"Waiting for participant in room {ctx.room.name}")
    participant = await utils.wait_for_participant(ctx.room)
    logger.info(f"Participant joined: {participant.identity}")
    
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(model="elevenlabs/eleven_flash_v2", language="en"),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    # Define data handler before registering it
    async def handle_data(dp: rtc.DataPacket):
        # Ignore messages from agent itself
        if dp.participant and dp.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return
        
        try:
            decoded_str = dp.data.decode("utf-8")
            try:
                data_json = json.loads(decoded_str)
                message_type = data_json.get("type")
                content = data_json.get("content")
            except json.JSONDecodeError:
                message_type = "raw"
                content = decoded_str
            
            if message_type == "job_description" and content:
                logger.info(f"Received JD via data channel: {content}...")
                # Acknowledge receipt
                await session.say("Job description received", allow_interruptions=False)
                
        except Exception as e:
            logger.error(f"Error handling data packet: {e}")
    
    def on_data_received(dp: rtc.DataPacket):
        asyncio.create_task(handle_data(dp))
    
    # Register the event handler
    ctx.room.on("data_received", on_data_received)
    
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    
    await session.say("Hello! paste your job description to get started.", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)