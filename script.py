from openai import OpenAI

from shaderrl.render import render_svg

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "user", "content": "Draw a red circle. Reply with only SVG code"}
    ],
    max_tokens=300,
    n=4,
)

for i, choice in enumerate(response.choices):
    text = choice.message.content
    print(f"--- answer {i} ---")
    print(text)

    image = render_svg(text)
    if image is None:
        print("could not render")
    else:
        image.save(f"out_{i}.png")
        print(f"saved out_{i}.png")