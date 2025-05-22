from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .host.gemini_chatbot import GeminiChatBot 
from fastapi.responses import StreamingResponse

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

# @app.post("/connect")
# async def connect(api_key: str):
#     try:
#         await chatbot.connect(api_key)
#         return 200
#     except Exception as e:
#         print(f"Error connecting to Gemini API: {str(e)}")
#         return 400

@app.post("/chat")
async def chat(request: Request):
    """Stream chat queries from the front-end."""
    data = await request.json()
    query = data.get("query", "")

    generator = chatbot.process_query(query)
    return StreamingResponse(generator, media_type="text/plain; charset=utf-8")