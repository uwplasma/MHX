# Writing style

MHX documentation uses short, direct technical English. A reader should find
the action, condition, and result on the first pass.

Use these rules:

1. Lead with the result or required action.
2. Use one term for each object.
3. Use common verbs such as `start`, `use`, `show`, and `check`.
4. Write in active voice.
5. Put a condition before its instruction.
6. Give one action in each numbered step.
7. Keep a sentence at 25 words or fewer.
8. Keep a paragraph at six sentences or fewer.
9. Use American spelling.
10. Do not use contractions, semicolons, or em dashes.

State measured facts with their settings and hardware. State a model limit
beside the related result. Do not replace evidence with claims about quality,
importance, novelty, or ease of use.

Comments explain a reason, constraint, unit, sign, or algorithm choice. They do
not repeat the code. Docstrings tell a beginner what an object does, what it
accepts, what it returns, and what can fail.

Run the prose check before a pull request:

```bash
python tools/check_prose.py
```

The check covers the README and the first-read documentation. It rejects common
filler terms, contractions, semicolons, em dashes, and long sentences.
