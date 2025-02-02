import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.ai.inference.models import SystemMessage, UserMessage

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_KEY"]),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Can you use python for creating a frontend app?"),
    ],
)

print("Response:", response.choices[0].message.content)
