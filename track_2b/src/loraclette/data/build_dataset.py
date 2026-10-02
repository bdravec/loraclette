"""Build the training data: ConGra -> prompt-completion jsonl.

TODO:
- prompt = [system, user] messages from loraclette.prompt; completion = the
  true resolution as an assistant message (TRL then trains on the completion only)
- write data/processed/train.jsonl and test.jsonl
- exclude the 3,597 python-tiny eval cases, ideally whole repos, and assert
  zero overlap before writing anything
"""
