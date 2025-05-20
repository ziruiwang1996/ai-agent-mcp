from dotenv import load_dotenv
import os
import sys
from google import genai
from google.genai import types
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from client.gemini_client import GeminiClient

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
            max_output_tokens=2048,
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
                    print(f"Calling tool {fc_name} with args {fc_args}")
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
                    print(response.text)
                    break
            else:
            # Pure text response—just print and exit
                print(response.text)
                break
        
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    

async def main():
    async with GeminiChatBot() as chatbot:
        await chatbot.chat_loop()

import asyncio
if __name__ == "__main__":
    asyncio.run(main())