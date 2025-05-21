from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
from contextlib import redirect_stdout, asynccontextmanager
from .host.gemini_chatbot import GeminiChatBot 

chatbot = GeminiChatBot()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the chatbot when the server starts."""
    await chatbot.__aenter__()
    yield
    """Clean up the chatbot when the server shuts down."""
    await chatbot.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    """Handle chat queries from the front-end."""
    data = await request.json()
    query = data.get("query", "")

    # Capture printed output
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            await chatbot.process_query(query)
    except Exception as e:
        return {"error": str(e)}

    # Return the captured output as the response
    output = buf.getvalue()
    return {"response": output}