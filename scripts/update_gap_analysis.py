import json
from pathlib import Path


def update_gap_analysis():
    repo_root = Path(__file__).parent.parent
    results_path = repo_root / "results" / "validation_results.json"
    gap_analysis_path = repo_root / "rust-numpy" / "GAP_ANALYSIS.md"

    if not results_path.exists():
        print("Error: validation_results.json not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    with open(gap_analysis_path, "r") as f:
        lines = f.readlines()

    # Find the Parity Table section
    table_start = -1
    for i, line in enumerate(lines):
        if "## Parity Table" in line:
            table_start = i
            break

    if table_start == -1:
        print("Error: Parity Table section not found in GAP_ANALYSIS.md")
        return

    # Keep headers and separators
    new_lines = lines[: table_start + 4]  # Assuming header + separator

    # Add results
    for res in results:
        row = f"| {res['module']} | {res['function']} | - | {res['dataset']} | {res['parity']} | {res['notes']} | {res['evidence']} |\n"
        new_lines.append(row)

    with open(gap_analysis_path, "w") as f:
        f.writelines(new_lines)

    print(f"Updated {gap_analysis_path}")


if __name__ == "__main__":
    update_gap_analysis()
