# Contributing to Prophet Forecasting Tool

Thank you for your interest in contributing! We welcome bug reports, feature requests, and code contributions.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone <your-fork-url>
    cd prophet_forecasting_tool
    ```
3.  **Create a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
4.  **Install dependencies in editable mode:**
    ```bash
    uv pip install -e .
    ```

## Development Workflow

1.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Make your changes.** Ensure you follow the existing coding style (PEP 8 recommendations).
3.  **Run Tests:**
    Run the unit test suite to ensure no regressions:
    ```bash
    uv run python -m unittest discover tests
    ```
4.  **Add Tests:** If you are adding a new feature, please include appropriate unit tests in the `tests/` directory.

## Web Front End Development

The Flask application is located in `src/prophet_forecasting_tool/app.py` with templates in `src/prophet_forecasting_tool/templates/`. The default table for display and forecasting is now `real_call_metrics`, with `ds` as the timestamp column and `y` as the value column.

To run the dev server:
```bash
uv run python src/prophet_forecasting_tool/app.py
```
Access it at `http://127.0.0.1:5001`.

## Submission

1.  **Push your branch** to GitHub:
    ```bash
    git push origin feature/my-new-feature
    ```
2.  **Open a Pull Request** against the `main` branch of the original repository.
3.  Provide a clear description of your changes and reference any related issues.
