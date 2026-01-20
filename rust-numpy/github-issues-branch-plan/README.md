# GitHub Issues -> Branch Plan Automation

This bundle contains two scripts:

1. `create_issues.sh`
   - Creates labels and issues for the NumPy->Rust port plan.

2. `create_project_board.sh`
   - Creates a GitHub Project (Projects v2), creates a `Stage` field, adds issues,
     and sets Stage.

## Quick start

From inside your repository:

```bash
gh auth login
gh auth refresh -s project

bash ./create_issues.sh

# After issues are created, add them to a Project board:
OWNER="@me" TITLE="NumPy->Rust Port" ISSUE_NUMBERS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" bash ./create_project_board.sh
```

If you want to operate on a specific repo without `cd`:

```bash
REPO="owner/repo" bash ./create_issues.sh
OWNER="@me" REPO="owner/repo" TITLE="NumPy->Rust Port" ISSUE_NUMBERS="..." bash ./create_project_board.sh
```
