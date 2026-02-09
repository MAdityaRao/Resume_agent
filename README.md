# Resume AI Voice Agent

A real-time AI voice interviewer built with LiveKit Agents that conducts professional technical interviews based on your resume and job descriptions.

## Overview

This voice AI agent acts as an intelligent interviewer, analyzing your resume against job descriptions and conducting realistic interview sessions. It maintains professional dialogue with sub-500ms latency and provides focused, concise responses.

## Features

- **Real-Time Voice Interaction**: Sub-500ms latency using LiveKit's voice infrastructure
- **Resume-Based Intelligence**: Answers strictly based on your resume content
- **Job Description Analysis**: Accepts JD input via data channel for targeted interview prep
- **Professional Tone**: Maintains business-appropriate, concise responses (30-45 words)
- **Noise Cancellation**: Enhanced audio quality with BVC noise cancellation
- **Preemptive Generation**: Faster response times with preemptive LLM generation

## Tech Stack

- **Voice Framework**: LiveKit Agents (Python)
- **Speech-to-Text**: Cartesia Ink Whisper
- **LLM**: OpenAI GPT-4o-mini
- **Text-to-Speech**: Cartesia Sonic Turbo
- **Voice Activity Detection**: Silero VAD
- **Noise Cancellation**: LiveKit BVC
- **Language**: Python 3.10+

## Prerequisites

- Python 3.10 - 3.13
- LiveKit Cloud account (or self-hosted LiveKit server)
- OpenAI API key
- Cartesia API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd resume_agent
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   pip install uv
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   
   Create a `.env.local` file in the project root:
   ```env
   LIVEKIT_URL=<your-livekit-url>
   LIVEKIT_API_KEY=<your-api-key>
   LIVEKIT_API_SECRET=<your-api-secret>
   OPENAI_API_KEY=<your-openai-key>
   CARTESIA_API_KEY=<your-cartesia-key>
   ```

4. **Add your resume**
   
   Place your resume content in `resume.txt` at the project root. The agent will load this file on startup.

## Usage

### Local Development

Run the agent locally:

```bash
python agent.py dev
```

The agent will:
1. Connect to your LiveKit room
2. Wait for a participant to join
3. Greet the participant and request job description
4. Conduct interview based on your resume and JD

### Sending Job Descriptions

Job descriptions can be sent to the agent via the data channel:

```javascript
// From your web client
const encoder = new TextEncoder();
const data = encoder.encode(JSON.stringify({
  type: "job_description",
  content: "Your job description text here..."
}));

await room.localParticipant.publishData(data, {
  reliable: true
});
```

### Production Deployment

The project includes a Dockerfile for containerized deployment:

```bash
docker build -t resume-agent .
docker run -e LIVEKIT_URL=<url> -e LIVEKIT_API_KEY=<key> resume-agent
```

Or deploy directly to LiveKit Cloud using the provided `livekit.toml` configuration.

## Project Structure

```
resume_agent/
├── agent.py              # Main agent logic
├── resume.txt            # Your resume content
├── pyproject.toml        # Python dependencies
├── uv.lock              # Locked dependencies
├── livekit.toml         # LiveKit configuration
├── Dockerfile           # Container configuration
├── .env.local           # Environment variables (create this)
└── README.md            # This file
```

## How It Works

1. **Initialization**: Agent loads resume content and initializes voice components
2. **Connection**: Establishes WebSocket connection to LiveKit room
3. **Session Start**: Waits for participant and starts voice session
4. **JD Processing**: Receives job description via data channel
5. **Interview Mode**: Responds to questions based on resume and JD
6. **Professional Responses**: Maintains 30-45 word limit, business tone

## Agent Instructions

The AI follows strict rules:
- Answers only from resume and provided JD
- Keeps responses to 30-45 words maximum
- Maintains professional, business-like tone
- Responds as if in a real interview
- Honestly states when JD doesn't match background
- Ignores non-interview questions

## Configuration

### Voice Models

Default configuration (in `agent.py`):
- **STT**: `cartesia/ink-whisper` (English)
- **LLM**: `openai/gpt-4o-mini`
- **TTS**: `cartesia/sonic-turbo` (English)

### Audio Options

- Noise cancellation enabled (BVC for regular calls, BVC Telephony for SIP)
- Preemptive generation for faster responses
- Voice activity detection with Silero VAD

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

The project uses Ruff for linting and formatting:

```bash
ruff check .
ruff format .
```

## Troubleshooting

**Resume not loading**
- Ensure `resume.txt` exists in the project root
- Check file encoding is UTF-8

**Connection issues**
- Verify `.env.local` contains correct LiveKit credentials
- Check network connectivity to LiveKit server

**Audio quality issues**
- Ensure stable internet connection
- Check microphone permissions
- Verify noise cancellation settings

## License

This project is available for personal and commercial use.

## Author

**Aditya**  
Full Stack Developer & AI Engineer  
- Portfolio: https://madityarao.github.io/Resume_web/
- GitHub: [@MAdityaRao](https://github.com/MAdityaRao)
- Email: madityara5@gmail.com

## Acknowledgments

- Built with [LiveKit Agents](https://docs.livekit.io/agents)
- Powered by OpenAI and Cartesia AI

---

**Live Demo**: Try the agent at https://madityarao.github.io/Resume/