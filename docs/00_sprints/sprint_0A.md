# Sprint 0A — Environment & Project Setup

## 1. Objective

The goal of this sprint is to establish a robust, reproducible, and scalable development environment for the project.

This includes setting up the project structure, development tooling, and ensuring that all components can be executed consistently across machines.

---

## 2. Project Structure

The project is organized following a modular and production-oriented structure:

```

project_root/
├── src/
├── scripts/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── tests/
├── configs/
└── README.md

```

**Rationale:**

- Separation between raw data, processed data, and metadata
- Clear distinction between scripts and core logic
- Scalable structure for future extensions (API, models, etc.)

---

## 3. Development Environment

### 3.1 Dev Container

A containerized development environment was configured using Docker and VS Code Dev Containers.

**Benefits:**

- Reproducibility across machines
- Isolation of dependencies
- Simplified onboarding

---

### 3.2 Python Environment

- Python version: 3.11
- Dependency management via pip (initial stage)

---

### 3.3 Code Quality

Pre-commit hooks were configured using:

- `ruff` (linting and formatting)

**Rationale:**

- Enforces consistent code style
- Prevents low-quality commits
- Reduces technical debt early

---

## 4. Version Control

- Git repository initialized
- Remote repository configured (GitHub)
- Standard commit structure adopted

---

## 5. Reproducibility

Key design principle:

> All steps in the pipeline must be executable from code, not manual actions.

This includes:
- Data download scripts
- Data validation scripts
- Dataset indexing

---

## 6. Outcome

At the end of this sprint:

- The development environment is fully reproducible
- The project structure is scalable and clean
- Code quality tools are integrated
- The repository is ready for collaborative and iterative development

---

## 7. Next Steps

Sprint 0B will focus on:
- Dataset acquisition
- BIDS validation
- Initial data exploration
```
