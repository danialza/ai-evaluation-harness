dataset = [  # Create the tiny exam dataset that our AI will be tested on.
    {"question": "What is 2 + 2?", "expected": "4"},  # Add question 1 and its correct answer.
    {"question": "What is 3 * 5?", "expected": "15"},  # Add question 2 and its correct answer.
    {"question": "If all cats are animals, are cats animals? yes or no", "expected": "yes"},  # Add question 3 and its correct answer.
    {"question": "What is 10 - 7?", "expected": "3"},  # Add question 4 and its correct answer.
]  # Finish the dataset list.


def make_prompt(question):  # Define a function that turns a plain question into a prompt for the AI.
    prompt = f"Answer only with the final answer.\nQuestion: {question}"  # Add a simple instruction before the question.
    return prompt  # Give the finished prompt back to the rest of the program.


def ask_model(prompt):  # Define the model connector, which is where a real AI call would usually happen.
    if "2 + 2" in prompt:  # Check whether the prompt contains the first math question.
        return "4"  # Return the toy model's answer for that question.
    if "3 * 5" in prompt:  # Check whether the prompt contains the second math question.
        return "15"  # Return the toy model's answer for that question.
    if "cats are animals" in prompt:  # Check whether the prompt contains the logic question.
        return "yes"  # Return the toy model's answer for that question.
    if "10 - 7" in prompt:  # Check whether the prompt contains the subtraction question.
        return "2"  # Return a wrong answer on purpose so we can see the grader catch it.
    return "I don't know"  # Return a fallback answer if the toy model does not recognize the prompt.


def grade_answer(model_answer, expected_answer):  # Define the grader that compares the AI answer to the answer key.
    clean_model_answer = model_answer.strip().lower()  # Clean the model answer by removing spaces and ignoring case.
    clean_expected_answer = expected_answer.strip().lower()  # Clean the expected answer the same way.
    is_correct = clean_model_answer == clean_expected_answer  # Check whether the cleaned answers match exactly.
    return is_correct  # Send True or False back to say whether the model passed this question.


correct_count = 0  # Start a counter for how many answers the model gets right.

for item in dataset:  # Loop through every question in the dataset, one at a time.
    question = item["question"]  # Pull the question text out of the current dataset item.
    expected = item["expected"]  # Pull the correct answer out of the current dataset item.
    prompt = make_prompt(question)  # Turn the raw question into the prompt we will send to the model.
    model_answer = ask_model(prompt)  # Send the prompt to the model connector and get the model's answer.
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
