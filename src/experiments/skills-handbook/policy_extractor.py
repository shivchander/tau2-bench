"""
Policy Extraction Engine

Uses OpenAI GPT models to extract structured policies from natural language
policy documents and map them to specific API functions and parameters.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class PolicyExtractor:
    """Extracts structured policies from policy documents using LLM."""

    def __init__(
        self,
        policy_path: str | Path,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
    ):
        """
        Initialize the policy extractor.

        Args:
            policy_path: Path to the policy document (markdown file)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            model: OpenAI model to use (default: "gpt-4.1")
            temperature: Sampling temperature 0.0-2.0 (default: 0.3 for more deterministic)
        """
        self.policy_path = Path(policy_path)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env file or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

        # Load policy document
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy document not found: {self.policy_path}")

        with open(self.policy_path, 'r') as f:
            self.policy_document = f.read()

        # Determine which prompt template to use based on policy path
        # Extract domain from policy path if possible
        self.domain = None
        if "airline" in str(policy_path).lower():
            self.domain = "airline"
        elif "retail" in str(policy_path).lower():
            self.domain = "retail"
        elif "telecom" in str(policy_path).lower():
            self.domain = "telecom"

        # Load domain-specific prompt template if available, otherwise use generic
        prompts_dir = Path(__file__).parent / "prompts"
        if self.domain:
            prompt_path = prompts_dir / f"{self.domain}_policy_extraction_prompt.txt"
            if not prompt_path.exists():
                # Fall back to generic
                prompt_path = prompts_dir / "policy_extraction_prompt.txt"
        else:
            prompt_path = prompts_dir / "policy_extraction_prompt.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

        if self.domain:
            print(f"  Using domain-specific prompt: {self.domain}")

        print(f"Initialized PolicyExtractor")
        print(f"  Policy document: {self.policy_path}")
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")

    def extract_policies(self, api_name: str, api_spec: list[dict]) -> dict:
        """
        Extract policies for an API from the policy document.

        Args:
            api_name: Name of the API (e.g., "airline")
            api_spec: List of function specifications (tau2 format)

        Returns:
            Structured policies dictionary
        """
        print(f"\nExtracting policies for {api_name} API...")
        print(f"  Found {len(api_spec)} functions in specification")

        # Prepare the prompt
        prompt = self._prepare_prompt(api_name, api_spec)

        # Call OpenAI
        try:
            print(f"  Calling OpenAI {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            # Extract JSON response
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from OpenAI")

            # Parse JSON
            policies = json.loads(response_text)
            print(f"  ✓ Successfully extracted policies")

            # Validate structure
            self._validate_policies(policies, api_spec)

            return policies

        except Exception as e:
            print(f"  ✗ Error extracting policies: {e}")
            raise

    def _prepare_prompt(self, api_name: str, api_spec: list[dict]) -> str:
        """Prepare the LLM prompt for policy extraction."""

        # Convert tau2 spec format to a more readable format for the LLM
        formatted_spec = []
        for func in api_spec:
            func_info = {
                "name": func.get("name", ""),
                "description": self._extract_description_from_doc(func.get("doc", "")),
                "parameters": func.get("params", {}).get("properties", {}),
                "required_parameters": func.get("params", {}).get("required", []),
            }
            formatted_spec.append(func_info)

        # Fill in the template
        prompt = self.prompt_template.format(
            policy_document=self.policy_document,
            api_spec=json.dumps(formatted_spec, indent=2),
            api_name=api_name,
        )

        return prompt

    def _extract_description_from_doc(self, doc: str) -> str:
        """Extract description from function docstring."""
        # The doc field contains the full function signature and docstring
        # Extract just the description part
        if '"""' in doc:
            parts = doc.split('"""')
            if len(parts) >= 2:
                return parts[1].strip().split('\n\n')[0].strip()
        return doc.split('\n')[0] if doc else ""

    def _validate_policies(self, policies: dict, api_spec: list[dict]) -> None:
        """Validate the extracted policies structure."""
        if "functions" not in policies:
            raise ValueError("Policies missing 'functions' key")

        # Ensure we have policies for each function (even if empty)
        spec_functions = {func["name"] for func in api_spec}
        policy_functions = {func["name"] for func in policies["functions"]}

        if not policy_functions.issubset(spec_functions):
            extra = policy_functions - spec_functions
            print(f"  Warning: Policies include unknown functions: {extra}")

        print(f"  Validated {len(policies['functions'])} function policies")

    def save_policies(self, policies: dict, output_path: str | Path) -> None:
        """Save extracted policies to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(policies, f, indent=2)

        print(f"  Saved policies to {output_path}")


def main():
    """Command-line interface for policy extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract structured policies from domain policy documents"
    )
    parser.add_argument(
        "domain",
        type=str,
        help="Domain name (e.g., airline, retail, telecom)",
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
        help="Sampling temperature 0.0-2.0 (default: 0.7 for deterministic extraction)",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        help="Custom path to policy.md file (overrides default domain path)",
    )
    parser.add_argument(
        "--spec-dir",
        type=str,
        help="Custom path to tool specifications directory (overrides default)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for policies (overrides default)",
    )

    args = parser.parse_args()

    # Setup paths based on domain
    base_dir = Path(__file__).parent
    repo_root = base_dir.parent.parent

    # Default paths
    if args.policy_path:
        policy_path = Path(args.policy_path)
    else:
        policy_path = repo_root / "data" / "tau2" / "domains" / args.domain / "policy.md"

    if args.spec_dir:
        spec_dir = Path(args.spec_dir)
    else:
        spec_dir = base_dir / "skills-handbook" / f"{args.domain}_spec"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "skills-handbook" / f"{args.domain}_policies"

    # Validate paths
    if not policy_path.exists():
        print(f"ERROR: Policy file not found: {policy_path}")
        print(f"Expected location: data/tau2/domains/{args.domain}/policy.md")
        print(f"Or specify custom path with --policy-path")
        return

    if not spec_dir.exists():
        print(f"ERROR: Spec directory not found: {spec_dir}")
        print(f"Expected location: src/experiments/skills-handbook/{args.domain}_spec/")
        print(f"Or specify custom directory with --spec-dir")
        print(f"\nTip: First run extract_{args.domain}_specs.py to generate tool specifications")
        return

    # Initialize extractor
    extractor = PolicyExtractor(
        policy_path=policy_path,
        model=args.model,
        temperature=args.temperature,
    )

    # Load all tool specs
    api_spec = []
    spec_files = sorted(spec_dir.glob("*.json"))

    if not spec_files:
        print(f"ERROR: No JSON spec files found in {spec_dir}")
        return

    for spec_file in spec_files:
        with open(spec_file, 'r') as f:
            api_spec.append(json.load(f))

    print(f"\nLoaded {len(api_spec)} tool specifications from {spec_dir}")

    # Extract policies
    policies = extractor.extract_policies(args.domain, api_spec)

    # Save policies
    output_path = output_dir / f"{args.domain}_policies.json"
    extractor.save_policies(policies, output_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Policy Extraction Summary - {args.domain.upper()}")
    print(f"{'='*60}")
    print(f"Total functions: {len(policies['functions'])}")

    # Count policies by function
    for func in policies['functions']:
        func_name = func['name']
        param_policies = func.get('policies', {})
        global_policies = func.get('global_policies', [])
        total_param_policies = sum(len(p) for p in param_policies.values())
        print(f"  {func_name}: {total_param_policies} parameter policies, {len(global_policies)} global policies")

    if 'cross_function_policies' in policies:
        print(f"\nCross-function policies: {len(policies['cross_function_policies'])}")

    print(f"\n✓ Policy extraction complete for {args.domain} domain")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
