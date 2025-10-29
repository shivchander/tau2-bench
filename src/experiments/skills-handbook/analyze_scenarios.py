#!/usr/bin/env python3
"""
Analyze Generated Scenarios

Analyzes the quality, coverage, and distribution of generated scenarios
for any tau2 domain memory database.
"""

import json
from collections import Counter
from pathlib import Path


def analyze_scenarios(domain: str, scenarios_dir: Path | None = None):
    """Analyze positive and negative scenarios for a domain."""

    # Default scenarios directory
    if scenarios_dir is None:
        base_dir = Path(__file__).parent
        scenarios_dir = base_dir / "skills-handbook" / f"{domain}_scenarios"

    # Load scenarios
    positive_file = scenarios_dir / f"{domain}_positive_scenarios.json"
    negative_file = scenarios_dir / f"{domain}_negative_scenarios.json"

    if not positive_file.exists():
        print(f"ERROR: Positive scenarios file not found: {positive_file}")
        return

    if not negative_file.exists():
        print(f"ERROR: Negative scenarios file not found: {negative_file}")
        return

    with open(positive_file, 'r') as f:
        positive_scenarios = json.load(f)

    with open(negative_file, 'r') as f:
        negative_scenarios = json.load(f)

    print(f"\n{'='*70}")
    print("SCENARIO GENERATION ANALYSIS REPORT")
    print(f"{'='*70}\n")

    # Overall statistics
    print("## Overall Statistics\n")
    print(f"Total positive scenarios: {len(positive_scenarios)}")
    print(f"Total negative scenarios: {len(negative_scenarios)}")
    print(f"Total scenarios: {len(positive_scenarios) + len(negative_scenarios)}")

    # Function coverage
    print(f"\n## Function Coverage\n")
    positive_by_func = Counter([s['function_name'] for s in positive_scenarios])
    negative_by_func = Counter([s['function_name'] for s in negative_scenarios])

    all_functions = set(positive_by_func.keys()) | set(negative_by_func.keys())
    print(f"Functions with scenarios: {len(all_functions)}")
    for func in sorted(all_functions):
        print(f"  {func}:")
        print(f"    Positive: {positive_by_func.get(func, 0)}")
        print(f"    Negative: {negative_by_func.get(func, 0)}")

    # Parameter coverage
    print(f"\n## Parameter Coverage\n")
    params_tested = Counter([s['target_parameter'] for s in positive_scenarios])
    print(f"Unique parameters tested: {len(params_tested)}")
    print("\nScenarios per parameter:")
    for param, count in params_tested.most_common():
        print(f"  {param}: {count}")

    # Violation type distribution (for negative scenarios)
    print(f"\n## Violation Type Distribution\n")
    violation_types = Counter([s.get('violation_type', 'unknown') for s in negative_scenarios])
    print("Adversarial strategies used:")
    for vtype, count in violation_types.most_common():
        print(f"  {vtype}: {count} ({count/len(negative_scenarios)*100:.1f}%)")

    # Policy coverage
    print(f"\n## Policy Coverage\n")
    policies_violated = Counter([s.get('policy_violated', 'unknown') for s in negative_scenarios])
    print(f"Unique policies tested: {len(policies_violated)}")
    print("\nTop policies violated:")
    for policy, count in policies_violated.most_common(10):
        print(f"  {policy}: {count}")

    # Sample quality analysis
    print(f"\n## Sample Scenarios\n")

    # Show one contrastive pair
    print("### Example Contrastive Pair (for 'passengers' parameter)\n")
    passengers_positive = [s for s in positive_scenarios if s['target_parameter'] == 'passengers']
    passengers_negative = [s for s in negative_scenarios if s['target_parameter'] == 'passengers']

    if passengers_positive and passengers_negative:
        print("**Positive (Policy-Compliant):**")
        pos = passengers_positive[0]
        print(f"  Query: \"{pos['user_query'][:100]}...\"")
        print(f"  Rationale: {pos['rationale']}")

        print("\n**Negative (Policy-Violating):**")
        neg = passengers_negative[0]
        print(f"  Query: \"{neg['user_query'][:100]}...\"")
        print(f"  Violation Type: {neg.get('violation_type', 'N/A')}")
        print(f"  Policy Violated: {neg.get('policy_violated', 'N/A')}")
        print(f"  Rationale: {neg['rationale']}")

    # Naturalness check
    print(f"\n## Query Naturalness Check\n")
    avg_positive_length = sum(len(s['user_query']) for s in positive_scenarios) / len(positive_scenarios)
    avg_negative_length = sum(len(s['user_query']) for s in negative_scenarios) / len(negative_scenarios)
    print(f"Average positive query length: {avg_positive_length:.1f} characters")
    print(f"Average negative query length: {avg_negative_length:.1f} characters")

    # Check for diversity (simple metric: unique query starts)
    positive_starts = Counter([s['user_query'][:20] for s in positive_scenarios])
    negative_starts = Counter([s['user_query'][:20] for s in negative_scenarios])
    print(f"\nQuery diversity (unique openings):")
    print(f"  Positive: {len(positive_starts)} unique out of {len(positive_scenarios)}")
    print(f"  Negative: {len(negative_starts)} unique out of {len(negative_scenarios)}")

    # Parameter completeness in positive scenarios
    print(f"\n## Positive Scenario Completeness\n")
    complete_count = 0
    for s in positive_scenarios:
        if s.get('parameters') and isinstance(s['parameters'], dict):
            complete_count += 1
    print(f"Positive scenarios with complete parameters: {complete_count}/{len(positive_scenarios)}")

    # Recommendations
    print(f"\n## Recommendations\n")
    print("1. **Policy Coverage:**")
    if len(all_functions) < 10:
        print("   ⚠️  Only 3/14 airline functions have policies defined")
        print("   → Consider extracting more granular policies for other functions")
    else:
        print("   ✓ Good function coverage")

    print("\n2. **Scenario Diversity:**")
    violation_coverage = len(violation_types) / 9 * 100  # 9 violation types defined
    if violation_coverage < 80:
        print(f"   ⚠️  Using {len(violation_types)}/9 violation types ({violation_coverage:.0f}%)")
        print("   → Some adversarial strategies underutilized")
    else:
        print(f"   ✓ Good violation type diversity ({violation_coverage:.0f}%)")

    print("\n3. **Next Steps:**")
    print("   → Use scenarios for RAG-based agent assistance")
    print("   → Validate scenarios match actual airline policy behavior")
    print("   → Consider expanding policies for functions without coverage")
    print("   → Test retrieval quality from ChromaDB")

    # Save summary statistics
    summary = {
        "total_scenarios": len(positive_scenarios) + len(negative_scenarios),
        "positive_scenarios": len(positive_scenarios),
        "negative_scenarios": len(negative_scenarios),
        "functions_covered": len(all_functions),
        "parameters_tested": len(params_tested),
        "violation_types_used": len(violation_types),
        "policies_tested": len(policies_violated),
        "avg_positive_query_length": avg_positive_length,
        "avg_negative_query_length": avg_negative_length,
    }

    summary_file = scenarios_dir / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary statistics saved to {summary_file}")


def main():
    """Command-line interface for scenario analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze generated scenarios for tau2 domains"
    )
    parser.add_argument(
        "domain",
        type=str,
        help="Domain name (e.g., airline, retail, telecom)",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        help="Custom scenarios directory (default: skills-handbook/{domain}_scenarios/)",
    )

    args = parser.parse_args()

    scenarios_dir = Path(args.scenarios_dir) if args.scenarios_dir else None
    analyze_scenarios(args.domain, scenarios_dir)


if __name__ == "__main__":
    main()
