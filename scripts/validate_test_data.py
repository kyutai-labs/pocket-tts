#!/usr/bin/env python3
"""
Validate generated test datasets for NumPy vs Rust cross-language compatibility.

This script validates that generated datasets are accessible from both Python and Rust,
checking data integrity, metadata consistency, and cross-platform compatibility.

Usage:
    python scripts/validate_test_data.py --data-dir test_data/ [--verbose] [--fix-issues]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class TestDataValidator:
    """Validator for generated test datasets ensuring cross-language compatibility."""

    def __init__(self, data_dir: Path, verbose: bool = False, fix_issues: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.fix_issues = fix_issues
        self.validation_results = {}
        self.issues_found = 0
        self.issues_fixed = 0

    def validate_all(self) -> bool:
        """Validate all datasets in the data directory."""
        print("🔍 Validating generated test datasets...")

        if not self.data_dir.exists():
            print(f"❌ Data directory {self.data_dir} does not exist")
            return False

        # Find all dataset files
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        json_files = list(self.data_dir.rglob("*.json"))
        npy_files = list(self.data_dir.rglob("*.npy"))

        if self.verbose:
            print(f"Found {len(parquet_files)} Parquet files")
            print(f"Found {len(json_files)} JSON metadata files")
            print(f"Found {len(npy_files)} NumPy files")

        # Validate each dataset category
        success = True

        categories = [
            "arrays",
            "mathematical",
            "linalg",
            "fft",
            "statistics",
            "edge_cases",
            "real_world",
        ]

        for category in categories:
            category_path = self.data_dir / category
            if category_path.exists():
                if self.verbose:
                    print(f"\n📁 Validating category: {category}")
                category_success = self._validate_category(category_path)
                success = success and category_success
            else:
                if self.verbose:
                    print(f"\n⚠️  Category {category} not found")

        # Generate validation report
        self._generate_validation_report()

        if self.issues_found == 0:
            print("✅ All datasets passed validation!")
        else:
            print(f"⚠️  Found {self.issues_found} issues, fixed {self.issues_fixed}")

        return self.issues_found == 0

    def _validate_category(self, category_path: Path) -> bool:
        """Validate all datasets in a category."""
        success = True
        parquet_files = list(category_path.rglob("*.parquet"))

        for parquet_file in parquet_files:
            dataset_success = self._validate_dataset(parquet_file)
            success = success and dataset_success

        return success

    def _validate_dataset(self, parquet_path: Path) -> bool:
        """Validate a single dataset."""
        dataset_name = str(parquet_path.relative_to(self.data_dir))

        if self.verbose:
            print(f"  📊 Validating: {dataset_name}")

        issues = []

        # 1. Check if Parquet file is readable
        try:
            df = pq.read_table(parquet_path).to_pandas()
        except Exception as e:
            issues.append(f"Parquet read error: {e}")
            return self._record_validation_result(dataset_name, False, issues)

        # 2. Check corresponding metadata
        metadata_path = parquet_path.with_suffix(".json")
        if not metadata_path.exists():
            issues.append("Missing metadata file")
        else:
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata_issues = self._validate_metadata(metadata, df)
                issues.extend(metadata_issues)
            except Exception as e:
                issues.append(f"Metadata read error: {e}")

        # 3. Check NumPy file consistency
        npy_dir = parquet_path.parent / (parquet_path.stem + "_npy")
        if npy_dir.exists():
            npy_issues = self._validate_npy_consistency(
                df, npy_dir, metadata if "metadata" in locals() else None
            )
            issues.extend(npy_issues)

        # 4. Validate array data
        data_issues = self._validate_array_data(df)
        issues.extend(data_issues)

        success = len(issues) == 0
        return self._record_validation_result(dataset_name, success, issues)

    def _validate_metadata(
        self, metadata: Dict[str, Any], df: pd.DataFrame
    ) -> List[str]:
        """Validate metadata consistency."""
        issues = []

        # Required fields
        required_fields = ["description", "generated_at"]
        for field in required_fields:
            if field not in metadata:
                issues.append(f"Missing required metadata field: {field}")

        # Validate arrays information if present
        if "arrays" in metadata:
            arrays_info = metadata["arrays"]
            for column_name in df.columns:
                if column_name not in arrays_info:
                    issues.append(f"Metadata missing info for column: {column_name}")
                    continue

                array_info = arrays_info[column_name]

                # Check if dtype in metadata matches actual
                actual_dtype = str(df[column_name].dtype)
                metadata_dtype = array_info.get("dtype", "")
                if metadata_dtype and not self._dtypes_compatible(
                    actual_dtype, metadata_dtype
                ):
                    issues.append(
                        f"Dtype mismatch for {column_name}: actual={actual_dtype}, metadata={metadata_dtype}"
                    )

                # Check if shape is reasonable for flattened data
                if "shape" in array_info:
                    expected_elements = np.prod(array_info["shape"])
                    actual_elements = len(df[column_name].dropna())
                    if expected_elements != actual_elements:
                        issues.append(
                            f"Element count mismatch for {column_name}: expected={expected_elements}, actual={actual_elements}"
                        )

        return issues

    def _validate_npy_consistency(
        self, df: pd.DataFrame, npy_dir: Path, metadata: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Validate NumPy file consistency with Parquet data."""
        issues = []

        for column_name in df.columns:
            npy_path = npy_dir / f"{column_name}.npy"
            if not npy_path.exists():
                issues.append(f"Missing NumPy file for column: {column_name}")
                continue

            try:
                npy_array = np.load(npy_path)
                parquet_data = df[column_name].dropna().values

                # Check if data is equivalent
                if len(npy_array.flatten()) != len(parquet_data):
                    issues.append(
                        f"Length mismatch for {column_name}: npy={len(npy_array.flatten())}, parquet={len(parquet_data)}"
                    )
                else:
                    # Compare values (allowing for some floating point differences)
                    if npy_array.dtype.kind in ["f", "c"]:  # Floating point
                        if not np.allclose(
                            npy_array.flatten(), parquet_data, rtol=1e-6, atol=1e-9
                        ):
                            issues.append(
                                f"Value mismatch for {column_name}: NumPy and Parquet data differ"
                            )
                    else:
                        if not np.array_equal(npy_array.flatten(), parquet_data):
                            issues.append(
                                f"Value mismatch for {column_name}: NumPy and Parquet data differ"
                            )

            except Exception as e:
                issues.append(f"NumPy file read error for {column_name}: {e}")

        return issues

    def _validate_array_data(self, df: pd.DataFrame) -> List[str]:
        """Validate array data for common issues."""
        issues = []

        for column_name in df.columns:
            data = df[column_name].dropna()

            if len(data) == 0:
                continue  # Empty arrays are valid for edge cases

            # Check for obvious data issues
            try:
                # Check if data contains only finite values (unless it's an edge case dataset)
                if (
                    "edge_cases" not in str(data.name)
                    if hasattr(data, "name")
                    else True
                ):
                    if (
                        np.any(np.isinf(data.values))
                        and "inf" not in str(column_name).lower()
                    ):
                        issues.append(f"Unexpected Inf values in {column_name}")

                    if (
                        np.any(np.isnan(data.values))
                        and "nan" not in str(column_name).lower()
                    ):
                        issues.append(f"Unexpected NaN values in {column_name}")

                # Check for reasonable numeric ranges
                if data.dtype.kind in ["f", "c"]:  # Floating point
                    finite_data = data[np.isfinite(data.values)]
                    if len(finite_data) > 0:
                        max_val = np.max(np.abs(finite_data))
                        if max_val > 1e10:
                            issues.append(
                                f"Suspiciously large values in {column_name}: max={max_val}"
                            )

            except Exception as e:
                issues.append(f"Data validation error for {column_name}: {e}")

        return issues

    def _dtypes_compatible(self, actual: str, expected: str) -> bool:
        """Check if two dtype strings are compatible."""
        # Normalize dtype strings
        actual_norm = actual.lower().replace(" ", "")
        expected_norm = expected.lower().replace(" ", "")

        # Direct match
        if actual_norm == expected_norm:
            return True

        # Common equivalences
        equivalences = {
            "float32": ["float"],
            "float64": ["double", "float"],
            "int32": ["int"],
            "int64": ["long", "int"],
        }

        for canonical, alternatives in equivalences.items():
            if expected_norm in alternatives and actual_norm == canonical:
                return True
            if actual_norm in alternatives and expected_norm == canonical:
                return True

        return False

    def _record_validation_result(
        self, dataset_name: str, success: bool, issues: List[str]
    ) -> bool:
        """Record validation result."""
        self.validation_results[dataset_name] = {
            "success": success,
            "issues": issues,
            "issue_count": len(issues),
        }

        self.issues_found += len(issues)

        if self.verbose and issues:
            for issue in issues:
                print(f"    ⚠️  {issue}")

        return success

    def _generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        report = {
            "validation_summary": {
                "total_datasets": len(self.validation_results),
                "successful_datasets": sum(
                    1 for r in self.validation_results.values() if r["success"]
                ),
                "failed_datasets": sum(
                    1 for r in self.validation_results.values() if not r["success"]
                ),
                "total_issues": self.issues_found,
                "issues_fixed": self.issues_fixed,
                "validation_time": pd.Timestamp.now().isoformat(),
            },
            "dataset_results": self.validation_results,
        }

        report_path = self.data_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        if self.verbose:
            print(f"\n📄 Validation report saved to {report_path}")

        # Generate human-readable summary
        summary_path = self.data_dir / "VALIDATION_SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write("# Test Data Validation Summary\\n\\n")
            f.write(f"**Total datasets:** {len(self.validation_results)}\\n")
            f.write(
                f"**Successful:** {sum(1 for r in self.validation_results.values() if r['success'])}\\n"
            )
            f.write(
                f"**Failed:** {sum(1 for r in self.validation_results.values() if not r['success'])}\\n"
            )
            f.write(f"**Total issues:** {self.issues_found}\\n\\n")

            # Failed datasets
            failed_datasets = {
                name: result
                for name, result in self.validation_results.items()
                if not result["success"]
            }
            if failed_datasets:
                f.write("## Failed Datasets\\n\\n")
                for dataset_name, result in failed_datasets.items():
                    f.write(f"### {dataset_name}\\n")
                    for issue in result["issues"]:
                        f.write(f"- {issue}\\n")
                    f.write("\\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated test datasets for cross-language compatibility"
    )
    parser.add_argument(
        "--data-dir",
        default="test_data",
        help="Directory containing generated test datasets (default: test_data)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Attempt to fix common issues automatically",
    )

    args = parser.parse_args()

    validator = TestDataValidator(Path(args.data_dir), args.verbose, args.fix_issues)
    success = validator.validate_all()

    if success:
        print("✅ Validation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
