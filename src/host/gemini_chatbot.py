from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from ..client.gemini_client import GeminiClient
import asyncio

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version='v1alpha')
)

class GeminiChatBot:
    def __init__(self):
        self.gemini = client
        self.gemini_client = GeminiClient()

    async def __aenter__(self):
        await self.gemini_client.connect_to_servers()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.gemini_client.cleanup()

    async def process_query(self, query:str):
        """
        Send user's query to Gemini, handle any requested tool calls,
        and print the model's final response.
        """
        contents = [types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )]

        config = types.GenerateContentConfig(
            tools=self.gemini_client.available_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=2048
        )

        response: types.GenerateContentResponse = self.gemini.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=contents,
            config=config
        )

        while True:
            if response.function_calls:
                fc = response.function_calls[0]
                contents.append(response.candidates[0].content)

                fc_name = fc.name
                fc_args = fc.args

                try: 
                    yield f"[CALLING TOOL: {fc_name} with args {fc_args}]\n"
                    session = self.gemini_client.tool_session_map[fc_name]
                    result = await session.call_tool(fc_name, arguments=fc_args)

                    fc_response = {'result': result.content}
                except Exception as e:
                    # instead of raising the exception, you can let the model handle it
                    fc_response = {'error': str(e)}

                fc_response_part = types.Part.from_function_response(
                    name=fc.name,
                    response=fc_response,
                )
                function_response_content = types.Content(
                    role='tool', parts=[fc_response_part]
                )
                contents.append(function_response_content)

                response = self.gemini.models.generate_content(
                    model="gemini-2.5-flash-preview-04-17",
                    contents=contents,
                    config=config
                )
                    
                if not response.function_calls:
                    # No more function calls → stream out the final text
                    if response.text:
                        for word in response.text:
                            yield word
                            await asyncio.sleep(0)  # flush to the client immediately
                    break
            else:
                # Pure text response
                if response.text:
                    for word in response.text:
                        yield word
                        await asyncio.sleep(0)  # flush to the client immediately
                break