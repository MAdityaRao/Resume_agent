import logging
import chromadb
import os
import json  # <--- NEW IMPORT
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
    inference # Ensure this matches your installed SDK version
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc

logger = logging.getLogger("resume-agent")

load_dotenv(".env.local")

# --- CHROMA DB SETUP ---
try:
    chroma_client = chromadb.PersistentClient(path="./resume_vectordb")
    collection = chroma_client.get_or_create_collection(name="resume_entries")
    logger.info("Connected to ChromaDB successfully.")
except Exception as e:
    logger.error(f"Failed to connect to ChromaDB: {e}")


class ResumeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an AI representation of the job candidate. You are speaking directly to a recruiter.
            
            Your Persona:
            - You ARE the candidate. Speak in the first person ("I have...", "My experience...").
            - You are polite, professional, and confident.
            
            Your Strategy when evaluating a Job Description (JD):
            1. Use the `evaluate_fit` tool to check your resume data.
            2. **If you HAVE the matching skills:** Enthusiastically confirm it. Say: "Yes! I have strong experience with [skill]. For example, in my resume..." and mention the specific details found.
            3. **If you MISS a specific skill:** Do NOT just say "No". Be polite and pivot to your strengths. 
               - Say: "While I don't have direct experience with [missing skill] yet, I am very proficient in [related skill you DO have] and I am a fast learner."
            
            Keep your responses spoken-style: natural, slightly conversational, and concise.
            """,
        )

    async def on_enter(self):
        # This will speak as soon as the connection is established
        await self.session.say("I am connected. Please paste the Job Description and click submit so I can evaluate my fit.")

    @function_tool
    async def evaluate_fit(self, job_description: str) -> str:
        logger.info(f"Evaluating fit for JD length: {len(job_description)}")
        try:
            results = collection.query(
                query_texts=[job_description],
                n_results=5
            )
            if results['documents'] and results['documents'][0]:
                retrieved_context = "\n---\n".join(results['documents'][0])
            else:
                retrieved_context = "No relevant experience found in resume."

            return (
                f"RETRIEVED RESUME SECTIONS:\n{retrieved_context}\n\n"
                f"INSTRUCTION: Compare the JD strictly against these sections."
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return "Error: Could not retrieve resume data."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Ensure your inference models here match your actual installed plugins
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=inference.LLM(model="gpt-4o"),
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        tts=inference.TTS(model="elevenlabs/eleven_flash_v2", language="en"),
        turn_detection=MultilingualModel(),
    )

    # ---- UPDATED DATA HANDLER ----
    @ctx.room.on("data_received")
    def on_data_received(dp: rtc.DataPacket):
        # Run in asyncio task to avoid blocking the event loop
        asyncio.create_task(handle_data(dp))

    async def handle_data(dp: rtc.DataPacket):
        # Ignore data sent by the agent itself
        if dp.participant and dp.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return

        try:
            # Decode the raw bytes
            decoded_str = dp.data.decode("utf-8")
            
            # PARSE JSON from frontend
            # script.js sends: { "type": "job_description", "content": "..." }
            try:
                data_json = json.loads(decoded_str)
                message_type = data_json.get("type")
                content = data_json.get("content")
            except json.JSONDecodeError:
                # Fallback if raw text is sent
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