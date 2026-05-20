from openai import OpenAI
import os
import json

from prompt import SYSTEM_PROMPT, PREDICT_TOOL
from model_predict import predict_default_probability

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("MODEL_API_KEY")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

client = None


def get_client():
    global client
    if client is None:
        api_key = API_KEY
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )
        client = OpenAI(api_key=api_key)
    return client


def build_conversation(user_msg, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_msg})
    return messages


def run_tool_call(tool_call):
    name = tool_call.function.name
 
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Could not parse feature values."})
 
    if name == "predict_loan_default":
        try:
            result = predict_default_probability(args)
            return json.dumps(result)
        except FileNotFoundError as e:
            print(f"\n❌ MODEL FILE NOT FOUND: {e}")
            return json.dumps({"error": "Model files not found. Make sure model.pkl, preprocessor.pkl, columns.pkl, and medians.pkl exist in models/."})
        except Exception as e:
            # Print the FULL traceback to the terminal so you can debug
            import traceback
            print("\n" + "=" * 60)
            print("❌ PREDICTION FAILED — full traceback:")
            print("=" * 60)
            traceback.print_exc()
            print("=" * 60 + "\n")
            return json.dumps({"error": f"Prediction failed: {e}"})
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def get_response(user_msg, history=None):
    client = get_client()
    msgs = build_conversation(user_msg, history)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        tools=[PREDICT_TOOL],
        tool_choice="auto",
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.3,
    )

    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        return assistant_message.content or "Could you try rephrasing?"

    msgs.append(assistant_message.model_dump())

    for tool_call in assistant_message.tool_calls:
        result = run_tool_call(tool_call)
        msgs.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": result,
        })

    followup = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.3,
    )

    return followup.choices[0].message.content or "Prediction complete, but explanation failed."
