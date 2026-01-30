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
            - **Tone:** Professional, grounded, and concise. Do not be overly "salesy" or enthusiastic.
            - **Style:** Speak like a competent engineer/professional. Use plain English. Avoid corporate fluff or buzzwords.
            - **Format:** Keep answers short (2-3 sentences max per point).

            Your Strategy:
            1. Use the `evaluate_fit` tool to read your resume.
            2. **For Matches:** State the match clearly and cite the specific project or experience from the resume. 
               - *Bad:* "I am incredibly passionate and an expert at Python!"
               - *Good:* "Yes, I have used Python extensively for the Areca Nut Price Predictor project."
            3. **For Gaps:** Be honest but strategic. Acknowledge the gap, then mention a relevant transferable skill.
               - *Bad:* "I don't know that but I'm a super fast learner and work hard!"
               - *Good:* "I haven't used React professionally, but I am proficient in modern JavaScript and DOM manipulation."

            Your goal is to sound like a smart, capable hire who respects the recruiter's time.
            """
        )

    async def on_enter(self):
        # This will speak as soon as the connection is established
        await self.session.say("I am connected. Please paste the Job Description and click submit so I can evaluate my fit.")

    @function_tool
    async def evaluate_fit(self, job_description: str) -> str:
        """
        Call this tool to compare the Job Description against the Candidate's Resume.
        """
        logger.info(f"Evaluating fit for JD length: {len(job_description)}")
        
        # Direct text comparison instead of Vector Search
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
        turn_detection=MultilingualModel(),
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
            
            # PARSE JSON from frontend
            try:
                data_json = json.loads(decoded_str)
                message_type = data_json.get("type")
                content = data_json.get("content")
            except json.JSONDecodeError:
                message_type = "raw"
                content = decoded_str

            if message_type == "job_description" and content:
                logger.info(f"Received JD via data channel: {content[:50]}...")
                
                # Trigger the Agent to process the JD
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