import os
import re


def scan_rust_codebase(src_path):
    inventory = []

    # Walk through the directory
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(".rs"):
                file_path = os.path.join(root, file)
                module_name = (
                    os.path.relpath(file_path, src_path)
                    .replace("/", "::")
                    .replace(".rs", "")
                )
                if module_name.endswith("::mod"):
                    module_name = module_name[:-5]

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Regex to find pub fn
                    # This is a simple regex and might miss some edge cases or complex logic,
                    # but should be good enough for a rough inventory.
                    # It captures: pub fn function_name
                    matches = re.findall(r"pub\s+fn\s+([a-z_][a-z0-9_]*)", content)

                    for func_name in matches:
                        inventory.append(
                            {
                                "module": module_name,
                                "function": func_name,
                                "status": "Implemented",  # Assumed implemented if found
                            }
                        )

    return inventory


def generate_markdown(inventory):
    md = "# Rust-NumPy Function Inventory\n\n"
    md += "| Rust Module | Rust Function | Status | Notes |\n"
    md += "|---|---|---|---|\n"

    # Sort for better readability
    inventory.sort(key=lambda x: (x["module"], x["function"]))

    for item in inventory:
        md += f"| {item['module']} | {item['function']} | {item['status']} | |\n"

    return md


if __name__ == "__main__":
    src_dir = "./src"
    inventory = scan_rust_codebase(src_dir)
    markdown_output = generate_markdown(inventory)

    # Print to stdout so we can capture it
    print(markdown_output)
