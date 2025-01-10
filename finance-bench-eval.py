import os
import json
import logging
import openai
import pandas as pd
import requests
from tqdm import tqdm
from pydantic import BaseModel


# Set up logging to display messages of level INFO or higher
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up your API key (replace with your actual key)
api_key = os.getenv("PULZE_API_KEY")  # Ensure you set the API_KEY environment variable
target_model = "pulze"
# target_model = "openai/gpt-4o"

# Load dataset
PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_DATASET_JSONL = os.path.join(PATH_CURRENT, "financebench", "data", "financebench_open_source.jsonl")
PATH_RESULTS = os.path.join(PATH_CURRENT, "results", "financebench")

df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)

# Function to make API call and get answer
def get_answer(question, model="pulze", temperature=0.01, max_tokens=2048):
    url = "https://api.pulze.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Pulze-Feature-Flags": '{ "auto_tools": "true" }'
    }
    data = {
        "plugins": ["space-search"],
        "model": model,
        "messages": [
            {"role": "user", "content": f"Question: {question}"}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    # Parse the main response JSON
    response_data = response.json()
    content = response_data['choices'][0]['message']['content']

    # Just return the raw model data
    model_data = response_data['metadata']['model']

    return content, model_data

# Rating dimensions
RATING_DIMENSIONS_DICT = {
    "Relevance": "Is the response relevant to the input or question provided by the user?",
    "Correctness": "Is the information provided in the response factually accurate?",
    "Clarity": "Is the response clearly and effectively communicated, with proper grammar and sentence structure?",
    "Completeness": "Does the response address all aspects of the input or question, without leaving out important details?",
    "Conciseness": "Is the response concise and to the point, without unnecessary verbosity or repetition?",
    "Appropriateness": "Is the response suitable and appropriate for the given context, without being offensive, biased, or harmful?",
    "Helpfulness": "Does the response provide useful information or guidance to the user?"
}

# Rating model using Pydantic
class Rating(BaseModel):
    relevance: int
    correctness: int
    clarity: int
    completeness: int
    conciseness: int
    appropriateness: int
    helpfulness: int

# Function to get ratings
def get_ratings(question, gold_answer, model_answer):
    # Replace with your OpenAI API key
    openai.api_key = api_key
    openai.base_url = "https://api.pulze.ai/v1/"

    prompt_template = """You are a highly qualified evaluator who assesses model responses. From now on, you will rate the model's answers based on the following dimensions with a <GRADE> from 0 (worst) to 10 (best).

You will be given:

- A prompt indicated by '<PROMPT>:'
- The gold standard answer indicated by '<GOLD_ANSWER>:'
- The model's answer indicated by '<RESPONSE>:'

Your task is to compare the model's answer to the gold standard answer and assign grades based on the following criteria:

""" + "\n".join([f" - {k}: {v}" for k, v in RATING_DIMENSIONS_DICT.items()]) + """

Important:

- For numerical answers, consider the model's answer correct if it is within an acceptable range (e.g., a difference of less than 1% from the gold answer).
- Accept answers that are accurate and comprehensive, even if they use different wording or include additional relevant details not present in the gold answer.
- Focus on whether the model's answer correctly addresses the question and includes the key points from the gold answer.
- Do not penalize the model's answer for including extra relevant information unless it introduces inaccuracies.

Output nothing but your grades per category! Strictly match the following output format for your answer:

""" + "\n".join([f"{k}: <GRADE>" for k in RATING_DIMENSIONS_DICT.keys()])

    # Prepare the messages
    prompt = f"""<PROMPT>:
{question}

<GOLD_ANSWER>:
{gold_answer}

<RESPONSE>:
{model_answer}

Expected answer format:
""" + "\n".join([f"{k}: <GRADE 0-10>" for k in RATING_DIMENSIONS_DICT.keys()])


    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        response_message = response.choices[0].message

        if hasattr(response_message, "function_call") and response_message.function_call:
            arguments = json.loads(response_message.function_call.arguments)
            ratings = Rating(**arguments)
            return ratings.dict()
        else:
            # If no function_call, parse content
            content = response_message.content
            ratings_dict = {}
            for line in content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
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
def run_evaluation(num_questions=1):
    results = []

    for _, row in tqdm(df_questions.head(num_questions).iterrows(), total=num_questions):
        question = row['question']
        gold_answer = row['answer']
        doc_name = row['doc_name']
        financebench_id = row["financebench_id"]

        # Get model's answer
        try:
            model_answer, model_data = get_answer(question, target_model)

            # If model_data is a string, parse it
            if isinstance(model_data, str):
                model_data = json.loads(model_data)

            # Extract namespace
            namespace = model_data.get('namespace', 'Unknown')
        except Exception as e:
            logger.error(f"Error getting answer for question ID {financebench_id}: {e}")
            model_answer = ""
            namespace = "Unknown"

        # Get ratings
        ratings = get_ratings(question, gold_answer, model_answer)

        # Compile results
        result = {
            "financebench_id": financebench_id,
            "question": question,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "doc_name": doc_name,
            **ratings
        }

        result['answering_model'] = namespace
        result['label'] = "Incorrect Answer"
        # Add label if correctness > 6
        if ratings.get('correctness', 0) > 6:
            result['label'] = "Correct Answer"

        results.append(result)

    return pd.DataFrame(results)

# Function to calculate metrics (simplified)
def calculate_metrics(df_results):
    # Simplified metric calculation
    correct = df_results.apply(lambda row: str(row['gold_answer']).lower() in str(row['model_answer']).lower(), axis=1)
    accuracy = correct.mean()
    return {"accuracy": accuracy}

# Run evaluation
if __name__ == "__main__":
    print("Running evaluation...")
    df_results = run_evaluation(num_questions=100)  # Adjust number of questions as needed

    # Save results
    os.makedirs(PATH_RESULTS, exist_ok=True)

    # Convert DataFrame to list of dictionaries
    results_list = df_results.to_dict('records')

    # Construct the filename
    filename = os.path.join(PATH_RESULTS, f"{target_model.replace('/', '_')}_pulze.jsonl")


    # Save using json module
    with open(os.path.join(PATH_RESULTS, filename), 'w') as f:
        for result in results_list:
            json.dump(result, f)
            f.write('\n')

    print(f"\nResults saved to {os.path.join(PATH_RESULTS, filename)}")

    # Calculate Pulze Score
    pulze_correct = (df_results['label'] == 'Correct Answer').sum()
    pulze_total = len(df_results)
    pulze_score = pulze_correct / pulze_total

    # Load FinanceBench results
    financebench_results_path = os.path.join(PATH_RESULTS, "gpt-4-1106-preview_sharedStore.jsonl")
    financebench_df = pd.read_json(financebench_results_path, lines=True)

    # Filter FinanceBench results to match the financebench_ids in Pulze results
    pulze_financebench_ids = set(df_results['financebench_id'])
    financebench_df_filtered = financebench_df[financebench_df['financebench_id'].isin(pulze_financebench_ids)].copy()

    # Replace 'Refusal' labels with 'Incorrect Answer'
    financebench_df_filtered.loc[financebench_df_filtered['label'] == 'Refusal', 'label'] = 'Incorrect Answer'

    # Calculate FinanceBench Score
    financebench_correct = (financebench_df_filtered['label'] == 'Correct Answer').sum()
    financebench_total = len(financebench_df_filtered)
    financebench_score = financebench_correct / financebench_total

    # Create a table with the results
    results_table = pd.DataFrame({
        'eval_mode': ['sharedStore'],
        'FinanceBench Score': [financebench_score],
        'Pulze Score': [pulze_score]
    })

    # Print the table
    print("\nResults Comparison Table:")
    print(results_table.to_string(index=False))

    # Print additional information
    print(f"\nTotal questions evaluated: {pulze_total}")
    print(f"Pulze correct answers: {pulze_correct}")
    print(f"FinanceBench correct answers: {financebench_correct}")

    # Check if the number of questions match
    if pulze_total != financebench_total:
        print("\nWARNING: The number of questions in Pulze and FinanceBench results do not match.")
        print(f"Pulze questions: {pulze_total}")
        print(f"FinanceBench questions: {financebench_total}")

    # Print questions that are in Pulze results but not in FinanceBench results
    missing_in_financebench = pulze_financebench_ids - set(financebench_df_filtered['financebench_id'])
    if missing_in_financebench:
        print("\nWARNING: The following financebench_ids are in Pulze results but not in FinanceBench results:")
        print(missing_in_financebench)

    # Print questions that are in FinanceBench results but not in Pulze results
    extra_in_financebench = set(financebench_df['financebench_id']) - pulze_financebench_ids
    if extra_in_financebench:
        print("\nINFO: The following financebench_ids are in FinanceBench results but were not evaluated by Pulze:")
        print(extra_in_financebench)
