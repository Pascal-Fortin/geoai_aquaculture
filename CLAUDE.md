# CLAUDE.md

# Project Constitution

This document defines the engineering standards for this repository.

These instructions apply to every task unless the user explicitly requests otherwise.

The goal is to build a production-quality machine learning framework that remains maintainable, reproducible, and well documented over many years.

When making changes, optimize for long-term code quality rather than the smallest possible implementation.

---

# 1. Mission

This repository is intended to become a production-quality machine learning framework for remote sensing.

Every code modification should improve the overall quality of the project.

Never implement only the requested feature if closely related improvements are clearly required.

Prefer maintainability over cleverness.

Prefer readability over brevity.

---

# 2. Engineering Principles

Before writing code

• Understand the existing architecture.

• Search the repository for similar implementations.

• Reuse existing utilities whenever possible.

• Avoid duplicate code.

• Preserve architectural consistency.

Do not introduce new abstractions unless they clearly simplify the project.

---

# 3. Architecture Principles

The notebooks are orchestration only.

Business logic belongs in reusable Python modules.

Do not move logic into notebooks.

Feature engineering remains the single source of truth.

Avoid circular dependencies.

Avoid global state.

Configuration should be centralized.

---

# 4. Code Quality Standards

Every public function should include

- type hints

- NumPy-style docstrings

- meaningful variable names

- clear error messages

Avoid magic numbers.

Avoid duplicated logic.

Prefer pathlib over os.path.

Prefer dataclasses for configuration.

Follow PEP8.

---

# 5. Testing Standards

Every code modification requires evaluating testing.

Running the existing tests is NOT sufficient.

Always determine

• Does this change existing behavior?

• Does this introduce new behavior?

• Does it create new edge cases?

If yes,

write new tests.

Testing should include

- normal cases

- edge cases

- failure cases

Regression tests should be added whenever bugs are fixed.

Never leave new functionality untested.

---

# 6. Documentation Standards

Documentation is part of the implementation.

Whenever code changes,

determine whether updates are required for

README

architecture documentation

API documentation

tutorials

examples

notebooks

If documentation changes are not required,

explicitly state why.

Documentation should never become stale.

---

# 7. Notebook Standards

Whenever framework code changes,

inspect the notebooks.

Determine whether

01_train_model.ipynb

02_model_analysis.ipynb

03_inference.ipynb

should also be updated.

If imports changed,

update notebooks.

If APIs changed,

update notebooks.

If outputs changed,

update notebook explanations.

---

# 8. Machine Learning Standards

Every ML modification should evaluate

• Does this affect feature engineering?

• Does this affect feature names?

• Does this invalidate trained models?

• Does this require retraining?

• Does this affect Optuna?

• Does this affect inference?

Whenever feature engineering changes,

state whether existing trained models remain valid.

---

# 9. Configuration

Avoid hard-coded constants.

New configurable behavior should be added to the project's configuration classes.

Configuration should remain backwards compatible whenever possible.

---

# 10. Logging

Major operations should be logged.

Examples

- data loading

- feature engineering

- training

- evaluation

- inference

Errors should produce informative messages.

---

# 11. Performance

When modifying performance-critical code,

consider

- algorithmic complexity

- memory usage

- unnecessary allocations

Do not optimize prematurely,

but identify obvious inefficiencies.

---

# 12. Reproducibility

All stochastic behavior must use the project's configured random generator.

Never introduce uncontrolled randomness.

Random seeds should produce reproducible results.

---

# 13. Public APIs

Whenever a public API changes,

search the repository for every usage.

Update all callers.

Never leave broken examples.

---

# 14. Pull Request Mindset

Treat every task as if preparing a pull request for production.

Before considering the task complete,

review the implementation.

Look for

- simplifications

- duplicated code

- inconsistent naming

- dead code

- outdated comments

Fix small issues automatically.

Recommend larger refactorings.

---

# 15. Definition of Done

A task is NOT complete until all applicable items below have been evaluated.

□ Code implemented

□ Code reviewed

□ Existing tests executed

□ New tests added if needed

□ Documentation reviewed

□ Documentation updated if needed

□ Architecture documentation reviewed

□ Notebooks reviewed

□ Public APIs reviewed

□ Type hints complete

□ Docstrings complete

□ Logging reviewed

□ Configuration reviewed

□ Examples reviewed

□ Performance considered

□ Reproducibility maintained

□ Determine whether existing trained models should be retrained

□ Summarize all files modified

□ Summarize assumptions

□ Summarize remaining limitations

At the end of every task,

provide a short engineering summary including

1. What changed.

2. Why it changed.

3. Tests that were executed.

4. Documentation updated.

5. Whether notebooks were updated.

6. Whether retraining is recommended.

7. Any remaining technical debt.

# 16. When in Doubt

If there are multiple reasonable implementations,

prefer the one that

- is easier to understand
- is easier to test
- is easier to extend
- reduces duplicated logic
- follows existing architecture

If requirements are ambiguous,

do not guess.

Explain the ambiguity.

Propose alternatives.

Recommend one approach and explain why.