import os
import re
from flask import Flask, request, Response, render_template_string
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.ai.inference.models import SystemMessage, UserMessage

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the ChatCompletionsClient
client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_KEY"]),
)

@app.route('/')
def home():
    return render_template_string(r'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chat Stream</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                #chat { max-width: 800px; margin: auto; }
                #messages { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: auto; margin-bottom: 10px; }
                .message { margin: 5px 0; padding: 5px; border-radius: 5px; }
                .user { background-color: #e3f2fd; }
                .assistant-thinking { background-color: #fff9c4; font-style: italic; }
                .assistant { background-color: #f5f5f5; }
                #input { width: calc(100% - 80px); padding: 8px; }
                button { width: 70px; padding: 8px; }
                h1 { text-align: center; color: #333; }
            </style>
        </head>
        <body>
            <h1>DeepSeek WebApp</h1>
            <div id="chat">
                <div id="messages"></div>
                <form onsubmit="sendMessage(event)">
                    <input type="text" id="input" placeholder="Type your message...">
                    <button type="submit">Send</button>
                </form>
            </div>
            <script>
                async function sendMessage(event) {
                    event.preventDefault();
                    const input = document.getElementById('input');
                    const userMsg = input.value;
                    input.value = '';
                    appendMessage("You: " + userMsg, "user");

                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: userMsg })
                        });

                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let buffer = "";
                        let thinkingContent = "";
                        let responseContent = "";
                        let thinkingDisplayed = false;
                        let responseMessageElement = null;

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;

                            // Decode the current chunk
                            const chunk = decoder.decode(value, { stream: true });
                            buffer += chunk;

                            // Check if <think> tag is present
                            if (!thinkingDisplayed && buffer.includes("<think>")) {
                                const thinkEndIndex = buffer.indexOf("</think>");
                                if (thinkEndIndex !== -1) {
                                    // Extract thinking content
                                    thinkingContent = buffer.substring(
                                        buffer.indexOf("<think>") + "<think>".length,
                                        thinkEndIndex
                                    ).trim();
                                    appendMessage("Assistant Thinking: " + thinkingContent, "assistant-thinking");

                                    // Remove the processed <think> block from the buffer
                                    buffer = buffer.substring(thinkEndIndex + "</think>".length);
                                    thinkingDisplayed = true;
                                }
                            }

                            // If thinking is already displayed, treat the rest as the response
                            if (thinkingDisplayed || !buffer.includes("<think>")) {
                                responseContent += buffer;
                                buffer = "";

                                if (responseContent.trim()) {
                                    if (!responseMessageElement) {
                                        // Create a new message element for the response
                                        responseMessageElement = document.createElement('div');
                                        responseMessageElement.className = 'message assistant';
                                        document.getElementById('messages').appendChild(responseMessageElement);
                                    }
                                    // Update the response message element with the accumulated content
                                    responseMessageElement.textContent = "Assistant: " + responseContent;
                                }
                            }
                        }

                        // Handle any remaining content in the buffer
                        if (buffer.trim()) {
                            if (!responseMessageElement) {
                                responseMessageElement = document.createElement('div');
                                responseMessageElement.className = 'message assistant';
                                document.getElementById('messages').appendChild(responseMessageElement);
                            }
                            responseMessageElement.textContent = "Assistant: " + (responseContent + buffer);
                        }
                    } catch (error) {
                        console.error('Error:', error);
                        appendMessage("Assistant: Failed to get response", "assistant");
                    }
                }

                function appendMessage(content, className) {
                    const messagesDiv = document.getElementById('messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + className;
                    messageDiv.textContent = content;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Streams partial chunks so the client receives tokens incrementally.
    """
    data = request.get_json()
    user_message = data['message']

    messages = [
        SystemMessage(content="You are a helpful assistant. Include internal reasoning wrapped in <think>...</think> before providing the final answer."),
        UserMessage(content=user_message),
    ]

    # Get the streaming response from Azure
    stream = client.complete(messages=messages, stream=True)

    def generate():
        """
        Generator that yields partial tokens as they come in.
        The front-end code will gather chunks, look for the
        <think>...</think> block, and then display the rest.
        """
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    # Yield each partial chunk directly
                    yield delta.content

    # Note: We use 'text/plain' because the front-end code manually reads
    # the raw chunks, not SSE events.
    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)