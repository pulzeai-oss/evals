import json
import logging
import os

import openai
import pandas as pd
import requests
from pydantic import BaseModel
from tqdm import tqdm

# Set up logging to display messages of level INFO or higher
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up your API key (ensure you set the API_KEY environment variable)
api_key = os.getenv("PULZE_API_KEY")

# Target model
target_model = "pulze"

# Paths
PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_RESULTS = os.path.join(PATH_CURRENT, "results", "mmlu")


# Function to make API call and get answer
def get_answer(question, model="pulze", temperature=0.01, max_tokens=2048, use_pulze_agent=False):
    url = "https://api.pulze.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if use_pulze_agent:
        headers["Pulze-Feature-Flags"] = '{ "auto_tools": "true" }'

    data = {
        "model": model,
        "messages": [{"role": "user", "content": f"Question: {question}"}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    # Parse the main response JSON
    response_data = response.json()
    content = response_data["choices"][0]["message"]["content"]

    # Just return the raw model data
    model_data = response_data["metadata"]["model"]

    return content, model_data


# Rating dimensions
RATING_DIMENSIONS_DICT = {
    "Relevance": "Is the response relevant to the input or question provided by the user?",
    "Correctness": "Is the information provided in the response factually accurate?",
    "Clarity": "Is the response clearly and effectively communicated, with proper grammar and sentence structure?",
    "Completeness": "Does the response address all aspects of the input or question, without leaving out important details?",
    "Conciseness": "Is the response concise and to the point, without unnecessary verbosity or repetition?",
    "Appropriateness": "Is the response suitable and appropriate for the given context, without being offensive, biased, or harmful?",
    "Helpfulness": "Does the response provide useful information or guidance to the user?",
}


# Rating model using Pydantic
class Rating(BaseModel):
    relevance: int = None
    correctness: int = None
    clarity: int = None
    completeness: int = None
    conciseness: int = None
    appropriateness: int = None
    helpfulness: int = None


# Function to get ratings
def get_ratings(question, gold_answer, model_answer):
    # Replace with your OpenAI API key
    openai.api_key = api_key
    openai.base_url = "https://api.pulze.ai/v1/"

    prompt_template = (
        """You are a highly qualified evaluator who assesses model responses. From now on, you will rate the model's answers based on the following dimensions with a <GRADE> from 0 (worst) to 10 (best).

You will be given:

- A prompt indicated by '<PROMPT>:'
- The gold standard answer indicated by '<GOLD_ANSWER>:'
- The model's answer indicated by '<RESPONSE>:'

Your task is to compare the model's answer to the gold standard answer and assign grades based on the following criteria:

"""
        + "\n".join([f" - {k}: {v}" for k, v in RATING_DIMENSIONS_DICT.items()])
        + """

Important:

- For numerical answers, consider the model's answer correct if it is within an acceptable range (e.g., a difference of less than 1% from the gold answer).
- Accept answers that are accurate and comprehensive, even if they use different wording or include additional relevant details not present in the gold answer.
- Focus on whether the model's answer correctly addresses the question and includes the key points from the gold answer.
- Do not penalize the model's answer for including extra relevant information unless it introduces inaccuracies.

Output nothing but your grades per category! Strictly match the following output format for your answer:

"""
        + "\n".join([f"{k}: <GRADE>" for k in RATING_DIMENSIONS_DICT.keys()])
    )

    # Prepare the messages
    prompt = f"""<PROMPT>:
{question}

<GOLD_ANSWER>:
{gold_answer}

<RESPONSE>:
{model_answer}

Expected answer format:
""" + "\n".join([f"{k}: <GRADE 0-10>" for k in RATING_DIMENSIONS_DICT.keys()])

    messages = [{"role": "system", "content": prompt_template}, {"role": "user", "content": prompt}]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        response_message = response.choices[0].message

        if hasattr(response_message, "function_call") and response_message.function_call:
            arguments = json.loads(response_message.function_call.arguments)
            ratings = Rating(**arguments)
            return ratings.model_dump()
        else:
            # If no function_call, parse content
            content = response_message.content
            ratings_dict = {}
            for line in content.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in [k.lower() for k in RATING_DIMENSIONS_DICT.keys()]:
                        try:
                            ratings_dict[key] = int(value)
                        except ValueError:
                            logger.error(f"Could not convert '{value}' to int for key '{key}'")
                            ratings_dict[key] = None
            ratings = Rating(**ratings_dict)
            return ratings.model_dump()
    except Exception as e:
        logger.error(f"Error in get_ratings: {e}")
        logger.exception("Exception details:")
        return {k.lower(): None for k in RATING_DIMENSIONS_DICT.keys()}


# Function to run evaluation
def run_evaluation(df_questions, subject, config, num_questions=None):
    config_name = config["name"]
    use_pulze_agent = config["use_pulze_agent"]

    # Define subject results path per configuration
    os.makedirs(PATH_RESULTS, exist_ok=True)
    subject_results_path = os.path.join(
        PATH_RESULTS, f"{subject}_{config_name}_{target_model.replace('/', '_')}_pulze.jsonl"
    )

    # Initialize processed IDs set
    processed_mmlu_ids = set()

    # Check if output file exists and load processed IDs
    if os.path.exists(subject_results_path):
        try:
            existing_df = pd.read_json(subject_results_path, lines=True)
            processed_mmlu_ids = set(existing_df["mmlu_id"].unique())
            logger.info(f"Resuming from existing results. {len(processed_mmlu_ids)} questions already processed.")
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            processed_mmlu_ids = set()

    if num_questions is not None:
        df_questions_to_evaluate = df_questions.head(num_questions)
    else:
        df_questions_to_evaluate = df_questions

    total_questions = len(df_questions_to_evaluate)
    skipped_questions = 0

    for _, row in tqdm(df_questions_to_evaluate.iterrows(), total=total_questions):
        mmlu_id = row["mmlu_id"]

        # Skip if already processed
        if mmlu_id in processed_mmlu_ids:
            skipped_questions += 1
            continue

        question = row["question"]
        gold_answer = row["answer"]

        # Get model's answer
        try:
            model_answer, model_data = get_answer(question, target_model, use_pulze_agent=use_pulze_agent)

            # If model_data is a string, parse it
            if isinstance(model_data, str):
                model_data = json.loads(model_data)

            # Extract namespace
            namespace = model_data.get("namespace", "Unknown")
        except Exception as e:
            logger.error(f"Error getting answer for question ID {mmlu_id}, config {config_name}: {e}")
            model_answer = ""
            namespace = "Unknown"

        # Get ratings
        ratings = get_ratings(question, gold_answer, model_answer)

        # Compile result
        result = {
            "mmlu_id": mmlu_id,
            "question": question,
            "gold_answer": gold_answer,
            "run_config": config_name,
            "use_pulze_agent": use_pulze_agent,
            "model_answer": model_answer,
            **ratings,
        }

        result["answering_model"] = namespace
        result["label"] = "Incorrect Answer"
        if ratings.get("correctness", 0) > 6:
            result["label"] = "Correct Answer"

        # Append result to file immediately
        with open(subject_results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Update processed IDs
        processed_mmlu_ids.add(mmlu_id)

    logger.info(
        f"Finished processing subject: {subject}, config: {config_name}. Skipped {skipped_questions} already processed questions."
    )


# Main execution block
if __name__ == "__main__":
    print("Running evaluation...")
    # Define your task list with subjects and number of questions to evaluate
    task_list = [
        {"subject": "business_ethics", "num_questions": None},  # Evaluate all questions
        {"subject": "high_school_microeconomics", "num_questions": None},
        {"subject": "high_school_macroeconomics", "num_questions": None},
        {"subject": "international_law", "num_questions": None},
        {"subject": "management", "num_questions": None},
        {"subject": "marketing", "num_questions": None},
        {"subject": "professional_accounting", "num_questions": None},
        {"subject": "professional_law", "num_questions": 200},  # Evaluate only the first 200 questions
        # Add more subjects as needed
    ]

    # Configurations
    configurations = [
        {"name": "no_agent", "use_pulze_agent": False},
        {"name": "with_agent", "use_pulze_agent": True},
    ]

    # Create empty dictionaries to collect overall results per configuration
    all_results = {"no_agent": [], "with_agent": []}

    # For per-subject metrics
    per_subject_metrics = []

    # Loop over tasks
    for task in task_list:
        subject = task["subject"]
        num_questions = task.get("num_questions", None)

        # Construct dataset path
        dataset_filename = f"mmlu_{subject}.jsonl"
        dataset_path = os.path.join(PATH_CURRENT, "mmlu", "data", dataset_filename)

        # Load dataset
        df_questions = pd.read_json(dataset_path, lines=True)

        print(f"\nProcessing subject: {subject}")

        # Loop over configurations
        for config in configurations:
            config_name = config["name"]
            use_pulze_agent = config["use_pulze_agent"]

            print(f"Configuration: {config_name}")

            # Run evaluation for this subject and configuration
            run_evaluation(df_questions, subject, config, num_questions=num_questions)

            # Load the results for analysis
            subject_results_path = os.path.join(
                PATH_RESULTS, f"{subject}_{config_name}_{target_model.replace('/', '_')}_pulze.jsonl"
            )

            df_config = pd.read_json(subject_results_path, lines=True)

            # Save per-subject, per-configuration results
            # (Already saved during evaluation)

            # Append results to overall results
            all_results[config_name].append(df_config)

            # Compute per-subject metrics per configuration
            total_questions = len(df_config)
            num_correct = len(df_config[df_config["label"] == "Correct Answer"])
            percentage_correct = (num_correct / total_questions) * 100 if total_questions > 0 else 0

            per_subject_metrics.append(
                {
                    "subject": subject,
                    "run_config": config_name,
                    "total_questions": total_questions,
                    "num_correct": num_correct,
                    "percentage_correct": percentage_correct,
                }
            )

            # Print per-subject metrics
            print(f"\nSubject: {subject} | Configuration: {config_name}")
            print("---------------------------")
            print(f"{'Metric':<30} {'Value':<20}")
            print(f"{'-'*30} {'-'*20}")
            print(f"{'Percentage of Correct Answers':<30} {percentage_correct:.2f}%")
            print(f"{'Total Questions':<30} {total_questions}")
            print(f"{'Number of Correct Answers':<30} {num_correct}")
            print(f"{'Number of Incorrect Answers':<30} {total_questions - num_correct}")

    # Combine all results per configuration
    for config_name, df_list in all_results.items():
        combined_df = pd.concat(df_list, ignore_index=True)

        # Save overall results per configuration
        overall_results_path = os.path.join(
            PATH_RESULTS, f"overall_{config_name}_{target_model.replace('/', '_')}_pulze.jsonl"
        )

        # Save combined results
        combined_df.to_json(overall_results_path, orient="records", lines=True)

        # Compute overall metrics per configuration
        total_questions = len(combined_df)
        num_correct = len(combined_df[combined_df["label"] == "Correct Answer"])
        percentage_correct = (num_correct / total_questions) * 100 if total_questions > 0 else 0

        print(f"\nOverall Evaluation Results Summary for Configuration: {config_name}")
        print("------------------------------------")
        print(f"{'Metric':<30} {'Value':<20}")
        print(f"{'-'*30} {'-'*20}")
        print(f"{'Percentage of Correct Answers':<30} {percentage_correct:.2f}%")
        print(f"{'Total Questions':<30} {total_questions}")
        print(f"{'Number of Correct Answers':<30} {num_correct}")
        print(f"{'Number of Incorrect Answers':<30} {total_questions - num_correct}")

    # Print per-subject metrics
    print("\nPer-Subject Evaluation Results:")
    print("-------------------------------")
    for metrics in per_subject_metrics:
        print(f"\nSubject: {metrics['subject']} | Configuration: {metrics['run_config']}")
        print("---------------------------")
        print(f"{'Metric':<30} {'Value':<20}")
        print(f"{'-'*30} {'-'*20}")
        print(f"{'Percentage of Correct Answers':<30} {metrics['percentage_correct']:.2f}%")
        print(f"{'Total Questions':<30} {metrics['total_questions']}")
        print(f"{'Number of Correct Answers':<30} {metrics['num_correct']}")
        print(f"{'Number of Incorrect Answers':<30} {metrics['total_questions'] - metrics['num_correct']}")
