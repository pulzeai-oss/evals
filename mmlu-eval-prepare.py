import hashlib
import json
import os

from datasets import load_dataset
from tqdm import tqdm

task_list = [
    "high_school_european_history",
    "business_ethics",
    "clinical_knowledge",
    "medical_genetics",
    "high_school_us_history",
    "high_school_physics",
    "high_school_world_history",
    "virology",
    "high_school_microeconomics",
    "econometrics",
    "college_computer_science",
    "high_school_biology",
    "abstract_algebra",
    "professional_accounting",
    "philosophy",
    "professional_medicine",
    "nutrition",
    "global_facts",
    "machine_learning",
    "security_studies",
    "public_relations",
    "professional_psychology",
    "prehistory",
    "anatomy",
    "human_sexuality",
    "college_medicine",
    "high_school_government_and_politics",
    "college_chemistry",
    "logical_fallacies",
    "high_school_geography",
    "elementary_mathematics",
    "human_aging",
    "college_mathematics",
    "high_school_psychology",
    "formal_logic",
    "high_school_statistics",
    "international_law",
    "high_school_mathematics",
    "high_school_computer_science",
    "conceptual_physics",
    "miscellaneous",
    "high_school_chemistry",
    "marketing",
    "professional_law",
    "management",
    "college_physics",
    "jurisprudence",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_macroeconomics",
    "computer_security",
    "moral_scenarios",
    "moral_disputes",
    "electrical_engineering",
    "astronomy",
    "college_biology",
]


def create_mmlu_id(subject, question):
    # Combine subject and question text
    id_str = f"{subject}-{question}"
    # Create a SHA256 hash
    mmlu_id = hashlib.sha256(id_str.encode()).hexdigest()
    # Shorten the hash to first 8 characters for readability
    mmlu_id_short = mmlu_id[:8]
    return f"mmlu_id_{mmlu_id_short}"


# Ensure the output directory exists
output_dir = "mmlu/data"
os.makedirs(output_dir, exist_ok=True)

for subject in task_list:
    print(f"Processing subject: {subject}")
    # Load the MMLU dataset for the current subject
    dataset = load_dataset("cais/mmlu", subject)
    # Access the 'test' split
    test_dataset = dataset["test"]

    # Define the output file path, adjusting the subject name
    # Replace any spaces or special characters in subject for file naming
    safe_subject = subject.replace(" ", "_").replace("/", "_")
    output_file = os.path.join(output_dir, f"mmlu_{safe_subject}.jsonl")

    # Open the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        # Iterate over the dataset
        for item in tqdm(test_dataset, desc=f"Writing {subject}"):
            # Extract fields from MMLU dataset
            question = item["question"]

            # The 'choices' might be in 'answer_choices' or 'choices' depending on the dataset
            # Let's check which field is correct
            if "choices" in item:
                choices = item["choices"]
            elif "answer_choices" in item:
                choices = item["answer_choices"]
            else:
                raise ValueError("Choices not found in dataset item.")

            # The 'answer' field is expected to be a string representing an option label ('A', 'B', 'C', 'D')
            correct_answer = item["answer"]

            # Create mmlu_id
            mmlu_id = create_mmlu_id(subject, question)

            # Construct the formatted question
            # Prepend with the desired prompt
            subject_name = subject.replace("_", " ").title()
            formatted_question = f"### Question: {question} ### Choices:"
            # formatted_question = f"Answer the following {subject_name} multiple-choice question. ### Question: {question} ### Choices:"

            # Append the choices formatted as "0) choice1 1) choice2 2) choice3 3) choice4"
            choices_str = " ".join([f"{idx}) {choice.strip()}" for idx, choice in enumerate(choices)])
            formatted_question += " " + choices_str

            # Construct the data in standardized format
            data = {
                "mmlu_id": mmlu_id,
                "subject": subject,
                "question_type": "multiple-choice",
                "question_reasoning": "Knowledge and reasoning",
                "question": formatted_question,
                "answer": correct_answer,
                "choices": choices,
                "dataset_subset_label": "MMLU",
            }

            # Write the data as JSONL
            json_line = json.dumps(data, ensure_ascii=False)
            f_out.write(json_line + "\n")
