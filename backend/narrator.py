from openai import OpenAI
client = OpenAI()

EPOCH_DESCRIPTIONS = {
    13.2: 'the universe is only 500 million years old',
    10.0: '3 billion years have passed since the Big Bang',
    7.0: 'the universe is in its peak star-forming era',
    4.5: 'Earth is forming in the Milky Way',
    0.0: 'the present day'
}

def narrate(galaxy_class: str, epoch_gyr: float, features: dict) -> str:
    epoch_ctx = EPOCH_DESCRIPTIONS.get(
        min(EPOCH_DESCRIPTIONS, key=lambda k: abs(k-epoch_gyr)), ''
    )
    prompt = f"""
You are narrating a galaxy evolution visualization.
Galaxy type: {galaxy_class}
Lookback time: {epoch_gyr:.1f} billion years ({epoch_ctx})

Write exactly 2 plain-English sentences describing:
1) what this galaxy looks like right now at this epoch
2) what physical process is dominating this evolution
no jargon, no numbers, vivid, exciting langauge."""
    
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': 'prompt'}],
        max_tokens=120,
        temperature=0.7
    )
    return response.choices[0].message.content