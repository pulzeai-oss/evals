"""
Import and process the pulze/intent-v0.1-dataset from Hugging Face.

This script:
1. Loads the dataset from Hugging Face
2. Groups by prompt_category + prompt combinations
3. Determines the winning model for each group based on vote counts
4. Selects the shortest response from the winning model
5. Outputs deduplicated JSONL file
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm


def determine_winner_response(row: Dict) -> Tuple[str, str]:
    """
    Determine which response won based on the winner field.

    Args:
        row: Dataset row containing winner, response_a, response_b

    Returns:
        Tuple of (winning_model, winning_response)
    """
    winner = row["winner"]

    if winner == 0:  # model_a wins
        return "model_a", row["response_a"]
    elif winner == 1:  # model_b wins
        return "model_b", row["response_b"]
    elif winner == 2:  # tie, use model_a as tie-breaker
        return "model_a", row["response_a"]
    else:
        # Fallback for unexpected values
        return "model_a", row["response_a"]


def process_group(group_rows: List[Dict]) -> Dict:
    """
    Process a group of rows with the same prompt_category + prompt combination.

    Args:
        group_rows: List of rows with identical prompt_category + prompt

    Returns:
        Dictionary with the final processed result
    """
    # Count wins for each model
    model_a_wins = 0
    model_b_wins = 0

    # Track all winning responses for each model
    model_a_responses = []
    model_b_responses = []

    for row in group_rows:
        winning_model, winning_response = determine_winner_response(row)

        if winning_model == "model_a":
            model_a_wins += 1
            model_a_responses.append(winning_response)
        else:  # model_b
            model_b_wins += 1
            model_b_responses.append(winning_response)

    # Determine overall winner for this group
    if model_a_wins > model_b_wins:
        winning_responses = model_a_responses
        winning_model = "model_a"
    elif model_b_wins > model_a_wins:
        winning_responses = model_b_responses
        winning_model = "model_b"
    else:  # tie, use model_a as tie-breaker
        winning_responses = model_a_responses
        winning_model = "model_a"

    # Select the shortest response from the winning model
    if winning_responses:
        shortest_response = min(winning_responses, key=len)
    else:
        # Fallback - shouldn't happen but just in case
        shortest_response = group_rows[0]["response_a"]

    # Use the first row as template for other fields
    template_row = group_rows[0]

    return {
        "question": template_row["prompt"],
        "subject": template_row["prompt_category"],
        "answer": shortest_response,
        "stats": {
            "total_rows": len(group_rows),
            "model_a_wins": model_a_wins,
            "model_b_wins": model_b_wins,
            "winning_model": winning_model,
            "answer_length": len(shortest_response),
        },
    }


def main():
    """Main processing function."""
    print("Loading pulze/intent-v0.1-dataset from Hugging Face...")

    try:
        # Load the dataset
        dataset = load_dataset("pulze/intent-v0.1-dataset")

        # Get the train split (adjust if needed)
        if "train" in dataset:
            data = dataset["train"]
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using split: {split_name}")

        print(f"Loaded {len(data)} rows from dataset")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Group rows by prompt_category + prompt combination
    print("Grouping rows by prompt_category + prompt...")
    groups = defaultdict(list)

    for row in tqdm(data, desc="Processing rows"):
        # Create a unique key for each prompt_category + prompt combination
        key = (row["prompt_category"], row["prompt"])
        groups[key].append(row)

    print(f"Found {len(groups)} unique prompt_category + prompt combinations")

    # Process each group
    print("Processing groups and selecting best responses...")
    processed_results = []
    stats = {
        "total_groups": len(groups),
        "total_original_rows": len(data),
        "model_a_group_wins": 0,
        "model_b_group_wins": 0,
        "ties": 0,
        "avg_answer_length": 0,
    }

    for (prompt_category, prompt), group_rows in tqdm(groups.items(), desc="Processing groups"):
        result = process_group(group_rows)
        processed_results.append(result)

        # Update stats
        if result["stats"]["winning_model"] == "model_a":
            if result["stats"]["model_a_wins"] > result["stats"]["model_b_wins"]:
                stats["model_a_group_wins"] += 1
            else:
                stats["ties"] += 1
        else:
            stats["model_b_group_wins"] += 1

    # Calculate average answer length
    if processed_results:
        stats["avg_answer_length"] = sum(r["stats"]["answer_length"] for r in processed_results) / len(
            processed_results
        )

    # Create output directory
    output_dir = Path("pulze-v0.1/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group results by subject
    print("Grouping results by subject...")
    subject_groups = defaultdict(list)
    for result in processed_results:
        subject_groups[result["subject"]].append(result)

    print(f"Found {len(subject_groups)} unique subjects: {list(subject_groups.keys())}")

    # Write separate JSONL file for each subject
    output_files = []
    for subject, subject_results in subject_groups.items():
        # Clean subject name for filename (replace spaces and special chars with underscores)
        clean_subject = subject.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")
        # Use the naming pattern expected by generic discovery: {benchmark_id}_{subject}.jsonl
        output_file = output_dir / f"pulze-v0.1_{clean_subject}.jsonl"
        output_files.append(output_file)

        print(f"Writing {len(subject_results)} items to {output_file}...")

        with open(output_file, "w", encoding="utf-8") as f:
            for result in subject_results:
                # Write only the essential fields to JSONL
                jsonl_row = {"question": result["question"], "subject": result["subject"], "answer": result["answer"]}
                f.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Original rows: {stats['total_original_rows']:,}")
    print(f"Unique combinations: {stats['total_groups']:,}")
    print(f"Final deduplicated rows: {len(processed_results):,}")
    print(
        f"Reduction: {((stats['total_original_rows'] - len(processed_results)) / stats['total_original_rows'] * 100):.1f}%"
    )
    print()
    print("Group Winners:")
    print(f"  Model A wins: {stats['model_a_group_wins']:,}")
    print(f"  Model B wins: {stats['model_b_group_wins']:,}")
    print(f"  Ties (defaulted to A): {stats['ties']:,}")
    print()
    print(f"Average answer length: {stats['avg_answer_length']:.1f} characters")
    print(f"Output files created: {len(output_files)}")
    for output_file in output_files:
        print(f"  - {output_file}")
    print()

    # Show a few examples
    print("Sample results:")
    for i, result in enumerate(processed_results[:3]):
        print(f"\n{i+1}. Subject: {result['subject']}")
        print(f"   Question: {result['question'][:100]}{'...' if len(result['question']) > 100 else ''}")
        print(f"   Answer: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
        print(f"   Stats: {result['stats']['total_rows']} rows → {result['stats']['winning_model']} wins")


if __name__ == "__main__":
    main()
