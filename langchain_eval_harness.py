try:  # Start a small safety check so beginners get a helpful message if LangChain is missing.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel  # Import a fake chat model for local practice.
    from langchain_core.output_parsers import StrOutputParser  # Import a parser that turns the model message into plain text.
    from langchain_core.prompts import ChatPromptTemplate  # Import LangChain's prompt template helper.
except ModuleNotFoundError:  # Catch the error that happens when LangChain is not installed yet.
    print("LangChain is not installed yet.")  # Tell the user what went wrong in plain language.
    print("Run: python3 -m pip install -r requirements-langchain.txt")  # Show the exact install command.
    raise SystemExit(1)  # Stop the script because it cannot continue without LangChain.


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


model = FakeListChatModel(  # Create a fake LangChain chat model so this demo can run without an API key.
    responses=["4", "15", "yes", "2"]  # Give the fake model one answer per dataset question.
)  # Finish creating the fake model.


output_parser = StrOutputParser()  # Create a parser that converts the model response into a simple string.


chain = prompt_template | model | output_parser  # Connect the prompt, model, and parser into one LangChain pipeline.


def ask_model(question):  # Define the model connector function used by our harness.
    model_answer = chain.invoke({"question": question})  # Send the question through the LangChain pipeline.
    return model_answer  # Give the model answer back to the evaluation loop.


def grade_answer(model_answer, expected_answer):  # Define the grader that compares the AI answer to the answer key.
    clean_model_answer = model_answer.strip().lower()  # Clean the model answer by removing spaces and ignoring case.
    clean_expected_answer = expected_answer.strip().lower()  # Clean the expected answer the same way.
    is_correct = clean_model_answer == clean_expected_answer  # Check whether the cleaned answers match exactly.
    return is_correct  # Send True or False back to say whether the model passed this question.


correct_count = 0  # Start a counter for how many answers the model gets right.

for item in dataset:  # Loop through every question in the dataset, one at a time.
    question = item["question"]  # Pull the question text out of the current dataset item.
    expected = item["expected"]  # Pull the correct answer out of the current dataset item.
    model_answer = ask_model(question)  # Ask the LangChain-powered model connector for an answer.
    passed = grade_answer(model_answer, expected)  # Ask the grader whether the model answer is correct.
    if passed:  # Check whether the grader said this answer was correct.
        correct_count = correct_count + 1  # Add one point to the score.
    print("Question:", question)  # Show the question that was tested.
    print("Expected:", expected)  # Show the answer key for this question.
    print("Model answered:", model_answer)  # Show what the model returned.
    print("Correct?", passed)  # Show whether the grader marked it correct.
    print("-" * 40)  # Print a small divider so each result is easy to read.

total_questions = len(dataset)  # Count how many questions were in the dataset.
score = correct_count / total_questions  # Turn the number correct into a percentage-style score.

print("Final score:", score)  # Print the final score as a decimal, like 0.75.
print(f"Final score percent: {score * 100:.0f}%")  # Print the final score in a friendlier percent format.
