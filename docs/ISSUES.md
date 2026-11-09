# Issues & Resolutions Log: AlphaCheck

This log records blockers, dependency collisions, and platform-specific resolutions encountered during development.

## Entry 1: PowerShell Syntax & Environment Setup
- **Date**: 2026-11-09
- **Issue**: Shell operators `&&` and reserved characters `<` caused parser errors in Windows PowerShell 5.1 when executing chained command blocks.
- **Resolution**: Used explicit PowerShell scripts or `cmd.exe /c` for multi-command invocations and standardized on the isolated virtual environment `.venv\Scripts\python.exe`.

## Entry 2: Git Author Auto-Detection
- **Date**: 2026-11-09
- **Issue**: Initial `git commit` failed due to missing global `user.name` and `user.email`.
- **Resolution**: Configured repository-local Git identity (`Aryan Rathor <aryanrathor@users.noreply.github.com>`) matching previous workspace conventions.

## Entry 3: Remote Repository Initial Push
- **Date**: 2026-11-09
- **Issue**: First push attempt returned `remote: Repository not found` because the GitHub repository had just been created or needed upstream branch initialization.
- **Resolution**: Successfully linked and established upstream tracking to `https://github.com/Aryan174-alt/AlphaCheck.git` on `main`.

