# Contributing to Urban Scene Intelligence

Thank you for contributing! This document explains how to contribute, the preferred workflow, testing, and commit conventions.

1) Report issues
- Search existing issues before opening a new one.
- Provide: a clear title, steps to reproduce, expected vs actual behavior, environment (OS, Python version), and minimal repro (image or data file) if applicable.

2) Branching & workflow
- Use branch names with clear prefixes:
  - `feature/<short-description>` for new features
  - `bugfix/<short-description>` for bug fixes
  - `docs/<short-description>` for documentation
- Keep changes small and focused; open one PR per logical change.
- Rebase or squash locally to keep history tidy.

3) Development setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4) Tests
- Add unit tests for new functionality.
- Run tests locally before opening a PR:
```powershell
pytest -q
```

5) Code style & formatting
- Use consistent formatting (black / flake8 recommended). We don't enforce a formatter yet, but keep code readable.

6) Commit messages
- Use concise, conventional-style messages, e.g.:
  - `feat: add scene-graph conversion helper`
  - `fix: handle empty detections in app.py`
  - `docs: update README with testing instructions`

7) Pull requests
- Provide a clear description of what changed and why.
- Link relevant issues.
- Include testing steps and screenshots if applicable.

8) Adding large files / models
- Do not commit large model weights (e.g., `.pt`) directly. Use the `scripts/download_weights.ps1` helper in this repo to download weights locally, or configure `git-lfs` for model files.

9) Ask for help
- If you're unsure where to make a change, open an issue or ping a core maintainer for guidance.

Thanks for helping improve this project!
