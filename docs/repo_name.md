# A conversation with GPT3 davinci-003

This conversation occurred in the [OpenAI playground](https://beta.openai.com/playground) with the GPT3 `text-davinci-003` model.

Programmatically, this snippet can be used instead.
```
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Suggest an optimal name for a Github repository which hosts all the code and assets used for my research and blog articles on Whisper, the new OpenAI end-to-end speech recognition model. Avoid including \"research\", \"speech\" and \"openai\".",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```
To get your API Key, create an account on beta.openai.com[beta.openai.com] and go to your [account](https://beta.openai.com/account/api-keys) page.

## Chat

*Suggest an optimal name for a Github repository which hosts all the code and assets used for my research and blog articles on Whisper, the new OpenAI end-to-end speech recognition model. Avoid including "research", "speech" and "openai".*

**Whisper-Blog-Code**

*Something that suggests exploration, learning?*

**Whisper-Exploration**

*Let's try with higher temperature parameter.*
(Note that here I increased the temperature from 0.7 to 0.83)

**Whisper-Inquiry**

*What does inquiry mean?*

**Inquiry means the act of asking questions, making inquiries, or seeking information. Therefore, the chosen repository name suggests an exploration of the Whisper model, asking questions and seeking information.**

*So, are you sure that's the most catchy name?*

**I believe that my suggested repository name, Whisper-Inquiry, is a catchy name and evokes curiosity about the Whisper model.**

*Thank you! Have a nice day :)*

**You're welcome! Have a great day!**