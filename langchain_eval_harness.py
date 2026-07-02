import json  # Import JSON so we can save the evaluation results to a results.json file.
import os  # Import os so we can read environment variables like GEMINI_API_KEY.
from datetime import datetime, timezone  # Import datetime so the output file can show when the run happened.


def load_env_file():  # Define a tiny helper that loads simple KEY=value lines from a local .env file.
    if not os.path.exists(".env"):  # Check whether a .env file exists in this folder.
        return  # Do nothing if the user has not created a .env file.
    with open(".env", "r", encoding="utf-8") as file:  # Open the .env file as normal text.
        for line in file:  # Read the .env file one line at a time.
            clean_line = line.strip()  # Remove extra spaces and newlines from this line.
            if not clean_line or clean_line.startswith("#"):  # Skip blank lines and comments.
                continue  # Move to the next line.
            if "=" not in clean_line:  # Skip lines that are not KEY=value pairs.
                continue  # Move to the next line.
            key, value = clean_line.split("=", 1)  # Split the line into the variable name and value.
            key = key.strip()  # Clean extra spaces around the variable name.
            value = value.strip().strip('"').strip("'")  # Clean spaces and simple quote marks around the value.
            os.environ.setdefault(key, value)  # Add the variable only if it is not already set in the shell.


load_env_file()  # Load local .env values before reading model settings.


try:  # Start a small safety check so beginners get a helpful message if LangChain is missing.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel  # Import a fake chat model for local practice.
    from langchain_core.prompts import ChatPromptTemplate  # Import LangChain's prompt template helper.
except ModuleNotFoundError:  # Catch the error that happens when LangChain is not installed yet.
    print("LangChain is not installed yet.")  # Tell the user what went wrong in plain language.
    print("Run: python3 -m pip install -r requirements-langchain.txt")  # Show the exact install command.
    raise SystemExit(1)  # Stop the script because it cannot continue without LangChain.


RESULTS_FILE = "results.json"  # Choose the filename where the harness will save its final output.
MODEL_MODE = os.environ.get("AI_HARNESS_MODEL", "fake").strip().lower()  # Read which model connector to use: fake or gemini.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")  # Read the Gemini model name, with a simple default.


dataset = [  # Create the tiny exam dataset that our AI will be tested on.
    {"question": "What is 2 + 2?", "expected": "4"},  # Add question 1 and its correct answer.
    {"question": "What is 3 * 5?", "expected": "15"},  # Add question 2 and its correct answer.
    {"question": "If all cats are animals, are cats animals? yes or no", "expected": "yes"},  # Add question 3 and its correct answer.
    {"question": "What is 10 - 7?", "expected": "3"},  # Add question 4 and its correct answer.
]  # Finish the dataset list.


prompt_template = ChatPromptTemplate.from_messages(  # Create a reusable LangChain prompt template.
    [  # Start the list of chat messages that will become the prompt.
        ("system", "Answer only with the final answer."),  # Add the instruction message for the model.
        ("human", "Question: {question}"),  # Add the question message, with a slot named question.
    ]  # Finish the list of chat messages.
)  # Finish creating the prompt template.


def build_model():  # Define a helper that chooses either the fake model or Gemini.
    if MODEL_MODE == "gemini":  # Check whether the user asked for the Gemini connector.
        if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):  # Check whether any supported Gemini API key exists.
            print("Gemini mode needs GOOGLE_API_KEY or GEMINI_API_KEY.")  # Explain what is missing.
            print("Example: export GEMINI_API_KEY='your-api-key-here'")  # Show the beginner-friendly setup command.
            raise SystemExit(1)  # Stop the script because Gemini cannot run without an API key.
        try:  # Try importing the Gemini LangChain integration only when Gemini mode is used.
            from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Gemini chat model connector.
        except ModuleNotFoundError:  # Catch the error that happens when the Gemini package is not installed.
            print("Gemini support is not installed yet.")  # Explain what is missing.
            print("Run: python3 -m pip install -r requirements-langchain.txt")  # Show the exact install command.
            raise SystemExit(1)  # Stop the script because the Gemini connector is unavailable.
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=2)  # Create and return the real Gemini model connector.
    return FakeListChatModel(responses=["4", "15", "yes", "2"])  # Otherwise return the fake local model for practice.


def message_to_text(message):  # Define a helper that turns a LangChain model message into plain text.
    if isinstance(message, str):  # Check whether the model already returned a plain string.
        return message  # Return the string directly.
    content = getattr(message, "content", message)  # Get the message content if it exists.
    if isinstance(content, str):  # Check whether the content is a normal string.
        return content  # Return the normal string.
    if isinstance(content, list):  # Check whether the content is a list of text blocks.
        text_parts = []  # Start an empty list for text chunks.
        for block in content:  # Loop through each content block from the model.
            if isinstance(block, dict) and "text" in block:  # Check whether this block is a dictionary with text.
                text_parts.append(block["text"])  # Add the text from this block.
            elif isinstance(block, str):  # Check whether this block is already a string.
                text_parts.append(block)  # Add the string block.
        return "\n".join(text_parts)  # Join all text chunks into one answer string.
    return str(content)  # Fall back to a string version if the content shape is unexpected.


model = build_model()  # Build the selected model connector.


chain = prompt_template | model  # Connect the prompt and model into one LangChain pipeline.


def ask_model(question):  # Define the model connector function used by our harness.
    message = chain.invoke({"question": question})  # Send the question through the LangChain pipeline.
    model_answer = message_to_text(message).strip()  # Convert the LangChain message into clean plain text.
    return model_answer  # Give the model answer back to the evaluation loop.


def grade_answer(model_answer, expected_answer):  # Define the grader that compares the AI answer to the answer key.
    clean_model_answer = model_answer.strip().lower()  # Clean the model answer by removing spaces and ignoring case.
    clean_expected_answer = expected_answer.strip().lower()  # Clean the expected answer the same way.
    is_correct = clean_model_answer == clean_expected_answer  # Check whether the cleaned answers match exactly.
    return is_correct  # Send True or False back to say whether the model passed this question.


def save_results(model_name, results, correct_count, total_questions, score):  # Define a helper that writes the run results to disk.
    output = {  # Create one dictionary that contains the full evaluation report.
        "model": model_name,  # Save the name of the model or connector we tested.
        "run_at": datetime.now(timezone.utc).isoformat(),  # Save the current UTC time for this evaluation run.
        "total_questions": total_questions,  # Save how many questions were tested.
        "correct_count": correct_count,  # Save how many questions were answered correctly.
        "score": score,  # Save the decimal score, like 0.75.
        "score_percent": round(score * 100),  # Save the friendly percent score, like 75.
        "results": results,  # Save the per-question details.
    }  # Finish the output dictionary.
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:  # Open results.json for writing with normal UTF-8 text.
        json.dump(output, file, indent=2)  # Write the output dictionary as nicely formatted JSON.


correct_count = 0  # Start a counter for how many answers the model gets right.
results = []  # Start an empty list where we will store each question's result.

for item in dataset:  # Loop through every question in the dataset, one at a time.
    question = item["question"]  # Pull the question text out of the current dataset item.
    expected = item["expected"]  # Pull the correct answer out of the current dataset item.
    model_answer = ask_model(question)  # Ask the LangChain-powered model connector for an answer.
    passed = grade_answer(model_answer, expected)  # Ask the grader whether the model answer is correct.
    if passed:  # Check whether the grader said this answer was correct.
        correct_count = correct_count + 1  # Add one point to the score.
    result = {  # Create a small result record for this one question.
        "question": question,  # Save the question text.
        "expected": expected,  # Save the answer key.
        "model_answer": model_answer,  # Save what the model answered.
        "correct": passed,  # Save whether the grader marked it correct.
    }  # Finish the result record.
    results.append(result)  # Add this question's result to the full results list.
    print("Question:", question)  # Show the question that was tested.
    print("Expected:", expected)  # Show the answer key for this question.
    print("Model answered:", model_answer)  # Show what the model returned.
    print("Correct?", passed)  # Show whether the grader marked it correct.
    print("-" * 40)  # Print a small divider so each result is easy to read.

total_questions = len(dataset)  # Count how many questions were in the dataset.
score = correct_count / total_questions  # Turn the number correct into a percentage-style score.
model_label = GEMINI_MODEL if MODEL_MODE == "gemini" else "fake-langchain-model"  # Pick a friendly model name for the report.
save_results(model_label, results, correct_count, total_questions, score)  # Save the final report to results.json.

print("Final score:", score)  # Print the final score as a decimal, like 0.75.
print(f"Final score percent: {score * 100:.0f}%")  # Print the final score in a friendlier percent format.
print(f"Saved results to {RESULTS_FILE}")  # Tell the user where the JSON output was saved.
