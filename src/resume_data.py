"""Structured resume data, sourced from Aditya_Resume.docx. Keep this as
data, not prose — the LLM never sees this file's raw text directly; it only
ever sees what the persona.py tools choose to return."""

CONTACT = {
    "name": "Aditya",
    "title": "AI/ML Developer · LLM Systems · Voice AI · Conversational Agents",
    "email": "madityara5@gmail.com",
    "phone": "+91 7338078108",
    "location": "Belman, Karnataka",
    "github": "github.com/MAdityaRao",
    "portfolio": "madityarao.github.io/Resume_web",
}

# The exact, unvarying line spoken for generic "tell me about Aditya" /
# "what does he do" questions. This is spoken directly by the get_about
# tool (see persona.py) — never paraphrased by the LLM.
ABOUT_FIXED_REPLY = (
    "Aditya's a second-year B.Sc Data Analytics student who's already shipping "
    "production AI systems — LLM-powered voice agents handling live inbound calls, "
    "plus automated data pipelines on AWS. He works across the full stack: audio "
    "pipelines, LLM reasoning, database design, and cloud deployment. Want me to "
    "check a job description against his skills?"
)

SUMMARY = (
    "Second-year B.Sc Data Analytics student with real production experience "
    "building AI systems — from LLM-powered voice agents handling live inbound "
    "calls to automated data pipelines integrated with cloud infrastructure. "
    "Comfortable working across the full stack: audio pipelines (STT/TTS), LLM "
    "reasoning layers, database design, and AWS deployment. Focused on shipping "
    "things that work."
)

# category -> list of (skill, resume line it's evidenced by)
SKILLS: dict[str, list[tuple[str, str]]] = {
    "AI / LLM": [
        ("GPT-4", "AI / LLM: GPT-4, Claude API, LangGraph, LangChain, Prompt Engineering, Multi-Turn Dialogue"),
        ("Claude API", "AI / LLM: GPT-4, Claude API, LangGraph, LangChain, Prompt Engineering, Multi-Turn Dialogue"),
        ("LangGraph", "AI / LLM: GPT-4, Claude API, LangGraph, LangChain, Prompt Engineering, Multi-Turn Dialogue"),
        ("LangChain", "AI / LLM: GPT-4, Claude API, LangGraph, LangChain, Prompt Engineering, Multi-Turn Dialogue"),
        ("Prompt Engineering", "AI / LLM: GPT-4, Claude API, LangGraph, LangChain, Prompt Engineering, Multi-Turn Dialogue"),
        ("Multi-Turn Dialogue", "AI / LLM: GPT-4, Claude API, LangGraph, LangChain, Prompt Engineering, Multi-Turn Dialogue"),
    ],
    "Voice & Real-Time": [
        ("LiveKit Agents", "Voice & Real-Time: LiveKit Agents, WebRTC, Deepgram STT, Cartesia TTS, Silero VAD, Plivo"),
        ("WebRTC", "Voice & Real-Time: LiveKit Agents, WebRTC, Deepgram STT, Cartesia TTS, Silero VAD, Plivo"),
        ("Deepgram STT", "Voice & Real-Time: LiveKit Agents, WebRTC, Deepgram STT, Cartesia TTS, Silero VAD, Plivo"),
        ("Cartesia TTS", "Voice & Real-Time: LiveKit Agents, WebRTC, Deepgram STT, Cartesia TTS, Silero VAD, Plivo"),
        ("Silero VAD", "Voice & Real-Time: LiveKit Agents, WebRTC, Deepgram STT, Cartesia TTS, Silero VAD, Plivo"),
        ("Plivo", "Voice & Real-Time: LiveKit Agents, WebRTC, Deepgram STT, Cartesia TTS, Silero VAD, Plivo"),
    ],
    "Languages": [
        ("Python", "Languages: Python, Flask, FastAPI, JavaScript (basic)"),
        ("Flask", "Languages: Python, Flask, FastAPI, JavaScript (basic)"),
        ("FastAPI", "Languages: Python, Flask, FastAPI, JavaScript (basic)"),
        ("JavaScript", "Languages: Python, Flask, FastAPI, JavaScript (basic) — note: basic proficiency"),
    ],
    "Data & Databases": [
        ("Pandas", "Data & Databases: Pandas, NumPy, Scikit-learn, SQL, PostgreSQL, asyncpg"),
        ("NumPy", "Data & Databases: Pandas, NumPy, Scikit-learn, SQL, PostgreSQL, asyncpg"),
        ("Scikit-learn", "Data & Databases: Pandas, NumPy, Scikit-learn, SQL, PostgreSQL, asyncpg"),
        ("SQL", "Data & Databases: Pandas, NumPy, Scikit-learn, SQL, PostgreSQL, asyncpg"),
        ("PostgreSQL", "Data & Databases: Pandas, NumPy, Scikit-learn, SQL, PostgreSQL, asyncpg"),
        ("asyncpg", "Data & Databases: Pandas, NumPy, Scikit-learn, SQL, PostgreSQL, asyncpg"),
    ],
    "Infrastructure": [
        ("AWS", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
        ("EC2", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
        ("Lambda", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
        ("Git", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
        ("GitHub", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
        ("REST APIs", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
        ("Google Sheets API", "Infrastructure: AWS (EC2, Lambda), Git, GitHub, REST APIs, Google Sheets API"),
    ],
}

# Aliases: query term (lowercase) -> canonical skill name in SKILLS above
SKILL_ALIASES: dict[str, str] = {
    "js": "JavaScript",
    "sql": "SQL",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "voice ai": "LiveKit Agents",
    "voice agents": "LiveKit Agents",
    "voice agent": "LiveKit Agents",
    "llm": "GPT-4",
    "llms": "GPT-4",
    "gpt": "GPT-4",
    "gpt-4": "GPT-4",
    "claude": "Claude API",
    "telephony": "Plivo",
    "sip": "Plivo",
    "vad": "Silero VAD",
    "stt": "Deepgram STT",
    "tts": "Cartesia TTS",
    "aws lambda": "Lambda",
    "ec2": "EC2",
    "sheets": "Google Sheets API",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "ml": "Scikit-learn",
    "machine learning": "Scikit-learn",
}

EXPERIENCE = [
    {
        "role": "AI Developer Intern",
        "org": "Torq Designs, Karnataka",
        "dates": "Aug 2025 – Present",
        "bullets": [
            "Designed and deployed 3+ production AI agents across voice, workflow, and LLM automation — replacing manual hotel reservation processes end-to-end for hospitality clients",
            "Built real-time telephony pipelines combining live audio, LLM reasoning, and structured data capture, cutting per-booking handling time by roughly 80%",
            "Maintained and scaled multi-agent AWS infrastructure with secure session handling, supporting simultaneous client deployments in production",
        ],
    },
]

PROJECTS = [
    {
        "name": "Hotel Booking Voice Automation Agent",
        "stack": "Python, LiveKit, OpenAI, Google Sheets API",
        "bullets": [
            "Independently designed and shipped a fully autonomous inbound call agent — greets guests, holds multi-turn conversations, captures structured booking data, and confirms reservations with no human involvement",
            "Tuned the STT/TTS pipeline via LiveKit WebRTC to stay under 500ms end-to-end latency; intent extraction (name, dates, room type) held above 95% accuracy across test calls",
            "Automated all booking writes to Google Sheets, removing manual data entry entirely and delivering a complete ready-to-deploy hospitality solution",
        ],
    },
    {
        "name": "Voice AI Insurance Assistant",
        "stack": "Python, LiveKit Agents, PostgreSQL, Deepgram, OpenAI, Cartesia, Silero VAD",
        "bullets": [
            "Built a full STT → LLM → TTS call pipeline using LiveKit Agents with natural conversation pacing through Silero VAD — handles inbound insurance customer queries",
            "Designed a normalised PostgreSQL schema covering customers, policies, claims, and call logs; the agent pulls live caller data via asyncpg and builds a personalised system prompt before each response",
            "Used the @function_tool pattern for mid-call database lookups triggered by phone number — removing policy ID verification steps and identifying callers automatically",
        ],
    },
]

EDUCATION = {
    "degree": "B.Sc in Data Analytics",
    "institution": "Dr. N.S.A.M. First Grade College, Nitte (Deemed to be University)",
    "dates": "2024 – 2027 (Expected)",
    "coursework": [
        "Statistical Analysis",
        "Machine Learning",
        "Data Visualisation",
        "Python for Data Science",
        "Database Management",
    ],
}