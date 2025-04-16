# vtes-judge-ai

Vtes Judge AI is a api that allows you to ask questions about Vampire: The Eternal Struggle (VTES) trading card game.

## Run the Application with Local Profile

First, you need to create a `.env` file in the root directory of the project with the following content:

```
OPENAI_API_KEY=[your open ai key here]
```

Then, you can run the application with the following command:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```