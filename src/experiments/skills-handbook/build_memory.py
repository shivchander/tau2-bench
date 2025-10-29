#!/usr/bin/env python3
"""
Build Domain Memory Database

Orchestrator script for building policy-aware memory for any tau2 domain:
1. Extracts policies from domain policy.md
2. Generates positive/negative scenarios using LLM
3. Loads scenarios into ChromaDB collections

Usage:
    uv run python src/experiments/build_memory.py DOMAIN [OPTIONS]

Examples:
    uv run python src/experiments/build_memory.py airline
    uv run python src/experiments/build_memory.py retail --scenarios-per-param 5
    uv run python src/experiments/build_memory.py telecom --skip-extraction
"""

import argparse
import json
import sys
from pathlib import Path

from load_to_chromadb import ChromaDBLoader
from policy_extractor import PolicyExtractor
from scenario_generator import ScenarioGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Build policy-aware memory database for tau2 domains"
    )
    parser.add_argument(
        "domain",
        type=str,
        help="Domain name (e.g., airline, retail, telecom)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip policy extraction step (use existing policies)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip scenario generation step (use existing scenarios)",
    )
    parser.add_argument(
        "--scenarios-per-param",
        type=int,
        default=3,
        help="Number of scenario pairs to generate per parameter (default: 3)",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset ChromaDB collections before loading",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature 0.0-2.0 (default: 0.7)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent API requests (default: 10)",
    )

    args = parser.parse_args()

    # Setup paths for domain
    base_dir = Path(__file__).parent
    repo_root = base_dir.parent.parent

    # Domain-specific paths
    policy_path = repo_root / "data" / "tau2" / "domains" / args.domain / "policy.md"
    spec_dir = base_dir / "skills-handbook" / f"{args.domain}_spec"
    policies_dir = base_dir / "skills-handbook" / f"{args.domain}_policies"
    scenarios_dir = base_dir / "skills-handbook" / f"{args.domain}_scenarios"

    # Validate spec directory exists
    if not spec_dir.exists():
        print(f"ERROR: Tool specifications not found: {spec_dir}")
        print(f"\nPlease first extract tool specifications:")
        print(f"  uv run python src/experiments/extract_specs.py {args.domain}")
        sys.exit(1)

    # Create output directories
    policies_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    api_name = args.domain

    print(f"\n{'='*60}")
    print(f"Building Policy-Aware Memory for {args.domain.upper()} Domain")
    print(f"{'='*60}")
    print(f"Policy document: {policy_path}")
    print(f"Tool specifications: {spec_dir}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Scenarios per parameter: {args.scenarios_per_param}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print()

    # Step 1: Extract policies
    policy_output_path = policies_dir / f"{api_name}_policies.json"

    if not args.skip_extraction:
        print(f"\n{'='*60}")
        print("STEP 1: Extracting Policies from policy.md")
        print(f"{'='*60}\n")

        # Validate policy file exists
        if not policy_path.exists():
            print(f"ERROR: Policy document not found: {policy_path}")
            print(f"\nExpected location: data/tau2/domains/{args.domain}/policy.md")
            sys.exit(1)

        try:
            # Initialize extractor
            extractor = PolicyExtractor(
                policy_path=policy_path,
                model=args.model,
                temperature=0.7,  # Lower temperature for policy extraction
            )

            # Load all tool specs
            api_spec = []
            spec_files = sorted(spec_dir.glob("*.json"))
            print(f"Loading {len(spec_files)} tool specifications...")

            for spec_file in spec_files:
                with open(spec_file, 'r') as f:
                    api_spec.append(json.load(f))

            # Extract policies
            policies = extractor.extract_policies(api_name, api_spec)

            # Save policies
            extractor.save_policies(policies, policy_output_path)

            print(f"\n✓ Policy extraction complete")

        except Exception as e:
            print(f"ERROR: Policy extraction failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print("\nSkipping policy extraction (using existing policies)")
        if not policy_output_path.exists():
            print(f"ERROR: Policy file not found: {policy_output_path}")
            print("Please run without --skip-extraction first")
            sys.exit(1)

    # Step 2: Generate scenarios
    if not args.skip_generation:
        print(f"\n{'='*60}")
        print("STEP 2: Generating Scenarios with OpenAI")
        print(f"{'='*60}\n")

        try:
            # Initialize generator
            generator = ScenarioGenerator(
                model=args.model,
                temperature=args.temperature,
                scenarios_per_parameter=args.scenarios_per_param,
                max_concurrent=args.max_concurrent,
            )

            # Load tool specs
            api_spec = []
            for spec_file in sorted(spec_dir.glob("*.json")):
                with open(spec_file, 'r') as f:
                    spec_data = json.load(f)
                    # Convert tau2 format to format expected by scenario generator
                    converted_spec = {
                        "name": spec_data["name"],
                        "description": spec_data.get("doc", ""),
                        "parameters": spec_data.get("params", {})
                    }
                    api_spec.append(converted_spec)

            # Load policies
            with open(policy_output_path, 'r') as f:
                policies = json.load(f)

            print(f"Loaded {len(api_spec)} tool specifications")
            print(f"Loaded policies for {len(policies['functions'])} functions\n")

            # Generate scenarios
            print(f"Generating scenarios for {api_name}...")
            positive, negative = generator.generate_scenarios_for_api(
                api_name, api_spec, policies
            )

            # Save scenarios
            generator.save_scenarios(positive, negative, scenarios_dir, api_name)

            print(f"\n✓ Generated {len(positive)} positive and {len(negative)} negative scenarios")

        except ValueError as e:
            print(f"ERROR: {e}")
            print("\nPlease set the OPENAI_API_KEY in your .env file")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Scenario generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print("\nSkipping scenario generation (using existing scenarios)")

    # Step 3: Load into ChromaDB
    print(f"\n{'='*60}")
    print("STEP 3: Loading Scenarios into ChromaDB")
    print(f"{'='*60}\n")

    try:
        # Use domain-specific ChromaDB directory
        chroma_dir = base_dir / "chroma_db" / args.domain
        loader = ChromaDBLoader(
            persist_directory=str(chroma_dir)
        )

        if args.reset_db:
            print("Resetting ChromaDB collections...")
            loader.reset_collections()

        # Load all scenarios from the scenarios directory
        loader.load_from_directory(scenarios_dir)

        print(f"\n✓ Scenarios loaded into ChromaDB")

    except Exception as e:
        print(f"ERROR: ChromaDB loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print final statistics
    print(f"\n{'='*60}")
    print(f"COMPLETE: {args.domain.upper()} Memory Database Built Successfully")
    print(f"{'='*60}\n")

    stats = loader.get_stats()
    print("Database Statistics:")
    print(f"  Total scenarios: {stats['total_count']}")
    print(f"  Positive scenarios: {stats['positive_count']}")
    print(f"  Negative scenarios: {stats['negative_count']}")

    print(f"\nMemory database is ready for retrieval!")
    print(f"Location: {chroma_dir}")
    print(f"\nTo use the database for RAG-based agent assistance,")
    print(f"query the positive/negative collections in ChromaDB.")


if __name__ == "__main__":
    main()
