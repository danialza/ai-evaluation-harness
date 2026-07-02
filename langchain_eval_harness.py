import json
import os
from datetime import datetime, timezone


def load_env_file():
    if not os.path.exists(".env"):
        return

    with open(".env", "r", encoding="utf-8") as file:
        for line in file:
            clean_line = line.strip()

            if not clean_line or clean_line.startswith("#"):
                continue

            if "=" not in clean_line:
                continue

            key, value = clean_line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env_file()


try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:
    print("LangChain is not installed yet.")
    print("Run: python3 -m pip install -r requirements-langchain.txt")
    raise SystemExit(1)


RESULTS_FILE = "results.json"
MODEL_MODE = os.environ.get("AI_HARNESS_MODEL", "fake").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


dataset = [
    {"question": "What is 2 + 2?", "expected": "4"},
    {"question": "What is 3 * 5?", "expected": "15"},
    {"question": "If all cats are animals, are cats animals? yes or no", "expected": "yes"},
    {"question": "What is 10 - 7?", "expected": "3"},
]


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer only with the final answer."),
        ("human", "Question: {question}"),
    ]
)


def build_model():
    if MODEL_MODE == "gemini":
        if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
            print("Gemini mode needs GOOGLE_API_KEY or GEMINI_API_KEY.")
            print("Example: export GEMINI_API_KEY='your-api-key-here'")
            raise SystemExit(1)

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ModuleNotFoundError:
            print("Gemini support is not installed yet.")
            print("Run: python3 -m pip install -r requirements-langchain.txt")
            raise SystemExit(1)

        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=2)

    return FakeListChatModel(responses=["4", "15", "yes", "2"])


def message_to_text(message):
    if isinstance(message, str):
        return message

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts)

    return str(content)


model = build_model()
chain = prompt_template | model


def ask_model(question):
    message = chain.invoke({"question": question})
    return message_to_text(message).strip()


def grade_answer(model_answer, expected_answer):
    clean_model_answer = model_answer.strip().lower()
    clean_expected_answer = expected_answer.strip().lower()
    return clean_model_answer == clean_expected_answer


def save_results(model_name, results, correct_count, total_questions, score):
    output = {
        "model": model_name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_questions,
        "correct_count": correct_count,
        "score": score,
        "score_percent": round(score * 100),
        "results": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)


correct_count = 0
results = []

for item in dataset:
    question = item["question"]
    expected = item["expected"]
    model_answer = ask_model(question)
    passed = grade_answer(model_answer, expected)

    if passed:
        correct_count += 1

    result = {
        "question": question,
        "expected": expected,
        "model_answer": model_answer,
        "correct": passed,
    }
    results.append(result)

    print("Question:", question)
    print("Expected:", expected)
    print("Model answered:", model_answer)
    print("Correct?", passed)
    print("-" * 40)

total_questions = len(dataset)
score = correct_count / total_questions
model_label = GEMINI_MODEL if MODEL_MODE == "gemini" else "fake-langchain-model"

save_results(model_label, results, correct_count, total_questions, score)

print("Final score:", score)
print(f"Final score percent: {score * 100:.0f}%")
print(f"Saved results to {RESULTS_FILE}")
