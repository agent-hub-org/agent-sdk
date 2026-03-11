from groq import Groq

client = Groq()

model = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=False)

