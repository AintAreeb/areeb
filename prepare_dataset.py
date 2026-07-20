"""
Converts a {{user}} / {{char}} style dialogue transcript into a JSONL
fine-tuning dataset (ChatML-style messages format).

Usage:
    python prepare_dataset.py raw_dialogue.txt dataset.jsonl --system system_prompt.txt

Each training example = the conversation so far (system + prior turns)
plus the target assistant reply. This lets the model learn to continue
the lesson in-character/in-format given history.
"""

import json
import re
import argparse


def parse_transcript(path):
    """Split a transcript into a list of (role, text) turns."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on lines that start a new turn: "{{user}}:" or "{{char}}:"
    pattern = re.compile(r"^\{\{(user|char)\}\}:\s?", re.MULTILINE)
    parts = pattern.split(raw)

    # parts[0] is leading whitespace/junk before first match; drop it
    turns = []
    it = iter(parts[1:])
    for role, text in zip(it, it):
        role = "user" if role == "user" else "assistant"
        text = text.strip()
        if text:
            turns.append((role, text))
    return turns


def build_examples(turns, system_prompt, min_context=1):
    """
    For every assistant turn, create one training example consisting of
    the system prompt + all prior turns as context + that assistant
    turn as the target completion.
    """
    examples = []
    history = []
    for role, text in turns:
        if role == "assistant" and len(history) >= min_context:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(
                {"role": h_role, "content": h_text} for h_role, h_text in history
            )
            messages.append({"role": "assistant", "content": text})
            examples.append({"messages": messages})
        history.append((role, text))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="Path to raw {{user}}/{{char}} transcript")
    ap.add_argument("output", help="Path to write JSONL dataset")
    ap.add_argument(
        "--system",
        default=None,
        help="Path to a text file containing the system prompt to prepend to every example",
    )
    args = ap.parse_args()

    system_prompt = ""
    if args.system:
        with open(args.system, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

    turns = parse_transcript(args.transcript)
    examples = build_examples(turns, system_prompt)

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Parsed {len(turns)} turns -> wrote {len(examples)} training examples to {args.output}")


if __name__ == "__main__":
    main()
