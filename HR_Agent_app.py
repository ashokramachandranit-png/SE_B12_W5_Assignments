import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


async def main() -> None:
  # 1. Initialize the model client with gpt-4o-mini
  model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

  try:
    # 2. Create the AssistantAgent with the exact system message
    hr_assistant = AssistantAgent(
        name="hr_assistant",
        system_message="You are an HR assistant.",
        model_client=model_client,
    )

    # 3. Task prompt formatted as 5 engaging numbered points (1, 2, 3...)
    task_prompt = (
        "Write a 5-line job description for a Junior Python Developer. "
        "Format it strictly as a numbered list (1 to 5) with engaging and "
        "attractive points highlighting key skills, career growth, and team"
        " culture."
    )

    # 4. Stream response using Console
    await Console(hr_assistant.run_stream(task=task_prompt))

  finally:
    # 5. Explicitly close the model client to prevent hanging
    await model_client.close()


if __name__ == "__main__":
  asyncio.run(main())