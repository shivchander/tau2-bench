"""
Scenario Generation Engine

Uses OpenAI GPT models to generate positive (policy-compliant) and negative
(policy-violating) scenarios for each API function and parameter.
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()


class ScenarioGenerator:
    """Generates synthetic scenarios using OpenAI GPT models with async API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        scenarios_per_parameter: int = 3,
        max_concurrent: int = 10,
    ):
        """
        Initialize the scenario generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            model: OpenAI model to use (default: "gpt-4.1")
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            scenarios_per_parameter: Number of scenario pairs per parameter (default: 3)
            max_concurrent: Maximum concurrent API calls (default: 10)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env file or pass api_key parameter."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.scenarios_per_parameter = scenarios_per_parameter
        self.max_concurrent = max_concurrent

        # Prompt template will be loaded dynamically based on API domain
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.prompt_template = None  # Will be set per API

        print(f"Initialized async OpenAI scenario generator")
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  Scenarios per parameter: {scenarios_per_parameter}")
        print(f"  Max concurrent requests: {max_concurrent}")

    def generate_scenarios_for_api(
        self, api_name: str, api_spec: list[dict], policies: dict
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate scenarios for all functions in an API.

        Args:
            api_name: Name of the API
            api_spec: List of function specifications
            policies: Extracted policies for the API

        Returns:
            Tuple of (positive_scenarios, negative_scenarios)
        """
        # Load domain-specific prompt template if available
        domain_prompt = self.prompts_dir / f"{api_name}_scenario_generation_prompt.txt"
        generic_prompt = self.prompts_dir / "scenario_generation_prompt.txt"

        if domain_prompt.exists():
            with open(domain_prompt, 'r') as f:
                self.prompt_template = f.read()
            print(f"  Using domain-specific prompt: {api_name}")
        elif generic_prompt.exists():
            with open(generic_prompt, 'r') as f:
                self.prompt_template = f.read()
            print(f"  Using generic prompt template")
        else:
            raise FileNotFoundError(f"No prompt template found in {self.prompts_dir}")

        # Run async generation in event loop
        return asyncio.run(self._generate_scenarios_async(api_name, api_spec, policies))

    async def _generate_scenarios_async(
        self, api_name: str, api_spec: list[dict], policies: dict
    ) -> tuple[list[dict], list[dict]]:
        """Async implementation of scenario generation."""
        all_positive = []
        all_negative = []

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Collect all generation tasks
        # Create a dict for quick lookup of policies by function name
        policies_by_name = {func["name"]: func for func in policies["functions"]}

        tasks = []
        for func_idx, func_spec in enumerate(api_spec):
            func_name = func_spec.get("name", "")

            # Skip functions without policies
            if func_name not in policies_by_name:
                print(f"  Skipping {func_name} (no policies defined)")
                continue

            func_policies = policies_by_name[func_name]
            parameters = func_spec.get("parameters", {}).get("properties", {})

            for param_name in parameters.keys():
                task = self._generate_scenarios_for_parameter(
                    semaphore=semaphore,
                    api_name=api_name,
                    func_spec=func_spec,
                    func_policies=func_policies,
                    target_parameter=param_name,
                )
                tasks.append((func_name, param_name, task))

        # Execute all tasks concurrently
        print(f"  Generating scenarios for {len(tasks)} parameters concurrently...")
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

        # Process results
        for (func_name, param_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"    ✗ {func_name}.{param_name}: Error - {result}")
                continue

            positive, negative = result
            if positive and negative:
                all_positive.extend(positive)
                all_negative.extend(negative)
                print(f"    ✓ {func_name}.{param_name}: Generated {len(positive)} scenarios")
            else:
                print(f"    ✗ {func_name}.{param_name}: No scenarios generated")

        return all_positive, all_negative

    async def _generate_scenarios_for_parameter(
        self,
        semaphore: asyncio.Semaphore,
        api_name: str,
        func_spec: dict,
        func_policies: dict,
        target_parameter: str,
    ) -> tuple[list[dict], list[dict]]:
        """Generate scenario pairs for a specific parameter (async)."""

        # Prepare the prompt
        prompt = self._prepare_prompt(
            api_name=api_name,
            func_spec=func_spec,
            func_policies=func_policies,
            target_parameter=target_parameter,
        )

        # Call OpenAI with rate limiting
        async with semaphore:
            try:
                # Debug: Print first prompt for inspection
                if not hasattr(self, '_printed_sample_prompt'):
                    print("\n" + "="*80)
                    print("SAMPLE PROMPT (first parameter):")
                    print("="*80)
                    print(prompt)
                    print("="*80 + "\n")
                    self._printed_sample_prompt = True

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                # Extract JSON response
                response_text = response.choices[0].message.content
                if not response_text:
                    return [], []

                # Parse JSON
                scenario_data = json.loads(response_text)

                # Handle different possible JSON structures
                scenario_pairs = scenario_data.get("scenarios", scenario_data.get("scenario_pairs", []))

                if not scenario_pairs:
                    return [], []

            except Exception as e:
                raise Exception(f"API call failed: {e}")

        # Separate positive and negative scenarios
        positive_scenarios = []
        negative_scenarios = []

        for pair in scenario_pairs:
            # Extract positive scenario
            if "positive" in pair:
                positive = pair["positive"]
                positive["api_name"] = api_name
                positive["target_parameter"] = target_parameter
                positive_scenarios.append(positive)

            # Extract negative scenario
            if "negative" in pair:
                negative = pair["negative"]
                negative["api_name"] = api_name
                negative["target_parameter"] = target_parameter
                # Ensure parameters is None for negative scenarios
                if "parameters" not in negative:
                    negative["parameters"] = None
                negative_scenarios.append(negative)

        return positive_scenarios, negative_scenarios

    def _prepare_prompt(
        self,
        api_name: str,
        func_spec: dict,
        func_policies: dict,
        target_parameter: str,
    ) -> str:
        """Prepare the LLM prompt for scenario generation."""

        function_name = func_spec.get("name", "")
        function_description = func_spec.get("description", "")

        # Format function spec for readability
        func_spec_str = json.dumps(func_spec, indent=2)

        # Format policies for readability
        policies_str = json.dumps(func_policies, indent=2)

        # Fill in the template
        prompt = self.prompt_template.format(
            api_name=api_name,
            function_name=function_name,
            function_description=function_description,
            function_spec=func_spec_str,
            policies=policies_str,
            target_parameter=target_parameter,
            num_scenarios=self.scenarios_per_parameter,
        )

        return prompt

    def save_scenarios(
        self,
        positive_scenarios: list[dict],
        negative_scenarios: list[dict],
        output_dir: str | Path,
        api_name: str,
    ) -> None:
        """Save generated scenarios to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save positive scenarios
        positive_path = output_dir / f"{api_name}_positive_scenarios.json"
        with open(positive_path, 'w') as f:
            json.dump(positive_scenarios, f, indent=2)

        # Save negative scenarios
        negative_path = output_dir / f"{api_name}_negative_scenarios.json"
        with open(negative_path, 'w') as f:
            json.dump(negative_scenarios, f, indent=2)

        print(f"  Saved {len(positive_scenarios)} positive scenarios to {positive_path}")
        print(f"  Saved {len(negative_scenarios)} negative scenarios to {negative_path}")


def main():
    """Example usage of scenario generator."""
    # Paths
    api_spec_dir = Path(__file__).parent.parent / "data" / "multi_turn_func_doc"
    policies_dir = Path(__file__).parent / "policies"
    scenarios_dir = Path(__file__).parent / "scenarios"

    # Initialize async generator
    generator = ScenarioGenerator(
        model="gpt-4.1",
        temperature=0.7,
        scenarios_per_parameter=3,
        max_concurrent=10,
    )

    # Load one API spec for testing
    test_api = "vehicle_control"
    spec_path = api_spec_dir / f"{test_api}.json"
    policy_path = policies_dir / f"{test_api}_policies.json"

    # Load JSONL format (one JSON object per line)
    api_spec = []
    with open(spec_path, 'r') as f:
        for line in f:
            if line.strip():
                api_spec.append(json.loads(line))

    with open(policy_path, 'r') as f:
        policies = json.load(f)

    # Generate scenarios
    print(f"\nGenerating scenarios for {test_api}...")
    positive, negative = generator.generate_scenarios_for_api(
        test_api, api_spec, policies
    )

    # Save scenarios
    generator.save_scenarios(positive, negative, scenarios_dir, test_api)

    print(f"\nGenerated {len(positive)} positive and {len(negative)} negative scenarios")


if __name__ == "__main__":
    main()
