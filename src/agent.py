import logging
import os
import json
import asyncio
from dotenv import load_dotenv

# Set tokenizers parallelism before importing libraries that use it
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    function_tool,
    inference
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc

logger = logging.getLogger("resume-agent")

load_dotenv(".env.local")

# --- LOAD RESUME TEXT ---
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


class ResumeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=
            """
            You are the job candidate. You are speaking to a recruiter. 
            
            Your Persona:
            - **Tone:** Professional, grounded, and concise.
            - **Style:** Speak like a competent engineer. Use plain English.
            - **Format:** Keep answers short (2-3 sentences max per point).

            Your Strategy:
            1. Use the `evaluate_fit` tool to read your resume.
            2. **For Matches:** Cite the specific project from the resume.
            3. **For Gaps:** Acknowledge the gap, then mention a relevant transferable skill.
            """
        )

    # We use 'on_connect' to speak immediately when the agent joins
    async def on_connect(self, ctx: JobContext):
        logger.info("Agent connected. Sending greeting.")
        # This will play uninterrupted because of the strict turn_detection below
        await self.session.say("I am connected. Please paste the Job Description and click submit so I can evaluate my fit.")

    @function_tool
    async def evaluate_fit(self, job_description: str) -> str:
        logger.info(f"Evaluating fit for JD length: {len(job_description)}")
        return (
            f"MY RESUME CONTENT:\n{RESUME_CONTENT}\n\n"
            f"THE JOB DESCRIPTION:\n{job_description}\n\n"
            f"INSTRUCTION: Compare the JD strictly against my resume content above."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()



async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=inference.LLM(model="gpt-4o"),
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        tts=inference.TTS(model="elevenlabs/eleven_flash_v2", language="en"),
        
      
    )

    # ---- DATA HANDLER ----
    @ctx.room.on("data_received")
    def on_data_received(dp: rtc.DataPacket):
        asyncio.create_task(handle_data(dp))

    async def handle_data(dp: rtc.DataPacket):
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
                logger.info(f"Received JD via data channel: {content[:50]}...")
                await session.generate_reply(
                    user_input=f"Here is the Job Description I need you to evaluate: {content}. Am I a fit?"
                )
            
        except Exception as e:
            logger.error(f"Error handling data packet: {e}")


    await session.start(
        agent=ResumeAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))