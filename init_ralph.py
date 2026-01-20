import json
import os


def main():
    with open("issues.json", "r") as f:
        issues = json.load(f)

    # Sort by number
    issues.sort(key=lambda x: x["number"])

    # 1. Create specs/issues.md
    os.makedirs("specs", exist_ok=True)
    with open("specs/issues.md", "w") as f:
        f.write("# Imported GitHub Issues\n\n")
        for issue in issues:
            f.write(f"## Issue #{issue['number']}: {issue['title']}\n\n")
            f.write(f"{issue['body']}\n\n")
            f.write("---\n\n")

    # 2. Create IMPLEMENTATION_PLAN.md
    with open("IMPLEMENTATION_PLAN.md", "w") as f:
        f.write("# Implementation Plan\n\n")
        f.write("Generated from GitHub Issues via Ralph Playbook Initialization.\n\n")
        for issue in issues:
            # Check labels for priority (optional, currently just naive list)
            labels = [l["name"] for l in issue["labels"]]
            prio = "[Med]"
            if "high" in str(labels).lower() or "critical" in str(labels).lower():
                prio = "[High]"

            f.write(f"- [ ] #{issue['number']} {prio} {issue['title']}\n")

    print(f"Generated specs/issues.md and IMPLEMENTATION_PLAN.md with {len(issues)} issues.")


if __name__ == "__main__":
    main()
