import logging
import chromadb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import asyncio
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
    metrics,
    inference
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero, noise_cancellation,elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc

logger = logging.getLogger("resume-agent")

load_dotenv(".env.local")

# --- CHROMA DB SETUP ---
# Initialize the database client once so it's ready for the agent
# Ensure your resume data is in the 'resume_vectordb' folder
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
               - Always try to bridge the gap by highlighting the value you DO bring.
            
            Keep your responses spoken-style: natural, slightly conversational, and concise.
            """,
        )

    async def on_enter(self):
        # Initial greeting when the agent joins
        await self.session.say("I am ready. Please send the Job Description via chat.")

    @function_tool
    async def evaluate_fit(self, job_description: str) -> str:
        """
        Called when the user provides a Job Description (JD) to check for fit.
        This tool searches the vector database for relevant experience.

        Args:
            job_description: The full text of the job description.
        """
        logger.info(f"Evaluating fit for JD: {job_description[:50]}...")

        try:
            # 1. Query ChromaDB for the most relevant resume sections
            results = collection.query(
                query_texts=[job_description],
                n_results=5  # Retrieve top 5 matching chunks
            )

            # 2. Format the results
            if results['documents'] and results['documents'][0]:
                retrieved_context = "\n---\n".join(results['documents'][0])
            else:
                retrieved_context = "No relevant experience found in resume."

            # 3. Return context to the LLM for final judgment
            return (
                f"RETRIEVED RESUME SECTIONS:\n{retrieved_context}\n\n"
                f"INSTRUCTION: Compare the JD strictly against these sections.\n"
                f"If key requirements are missing, you MUST say 'No'."
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return "Error: Could not retrieve resume data. Please try again."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ---- ENTRYPOINT ----
async def entrypoint(ctx: JobContext):

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=inference.LLM(model="gpt-4o"),
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        tts=inference.TTS(model="elevenlabs/eleven_flash_v2", language="en"),
        turn_detection=MultilingualModel(),
    )

    # ---- DATA CHANNEL HANDLER ----
    import asyncio

    @ctx.room.on("data_received")
    def on_data_received(dp: rtc.DataPacket):
        asyncio.create_task(handle_data(dp))

    async def handle_data(dp: rtc.DataPacket):
        if dp.participant and dp.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return

        try:
            message_text = dp.data.decode("utf-8")
        except Exception as e:
            logger.error(e)
            return

        # ✅ SAFE: session already started
        await session.generate_reply(
            user_input=f"Here is the Job Description: {message_text}. Am I a fit?"
        )

    # ---- START AGENT FIRST ----
    await session.start(
        agent=ResumeAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # ---- CONNECT ROOM ----
    await ctx.connect()





if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))