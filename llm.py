from openai import OpenAI 

def spell_check(word):
    client = OpenAI(
        api_key="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"
    )
    client.models.list()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a spell checker. You give 5 examples of what a given given word could be. Give distinct examples in order of likelihood."},
        {"role": "user", "content": word}
    ]
    )
    string = completion.choices[0].message.content 
    return string