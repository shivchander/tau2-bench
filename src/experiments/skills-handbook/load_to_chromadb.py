"""
ChromaDB Loader

Loads positive and negative scenarios into ChromaDB collections for retrieval.
Uses OpenAI text-embedding-3-large for high-quality semantic embeddings.
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ChromaDBLoader:
    """Loads scenarios into ChromaDB collections with OpenAI embeddings."""

    def __init__(self, persist_directory: str | Path | None = None):
        """
        Initialize the ChromaDB loader with OpenAI embeddings.

        Args:
            persist_directory: Directory to persist ChromaDB data
                              (defaults to ./chroma_db in this module)
        """
        if persist_directory is None:
            persist_directory = Path(__file__).parent / "chroma_db"
        else:
            persist_directory = Path(persist_directory)

        persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=str(persist_directory))

        # Create OpenAI embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-large"
        )

        print(f"Using OpenAI text-embedding-3-large for embeddings")

        # Create or get collections with OpenAI embeddings
        self.positive_collection = self.client.get_or_create_collection(
            name="positive_scenarios",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Policy-compliant (happy path) scenarios",
                "embedding_model": "text-embedding-3-large"
            }
        )

        self.negative_collection = self.client.get_or_create_collection(
            name="negative_scenarios",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Policy-violating (adversarial) scenarios",
                "embedding_model": "text-embedding-3-large"
            }
        )

        print(f"ChromaDB initialized at: {persist_directory}")
        print(f"  Positive collection: {self.positive_collection.count()} documents")
        print(f"  Negative collection: {self.negative_collection.count()} documents")

    def load_scenarios(
        self,
        positive_scenarios: list[dict],
        negative_scenarios: list[dict],
    ) -> None:
        """
        Load scenarios into ChromaDB collections.

        Args:
            positive_scenarios: List of positive scenario dictionaries
            negative_scenarios: List of negative scenario dictionaries
        """
        print(f"\nLoading {len(positive_scenarios)} positive scenarios...")
        self._load_positive_scenarios(positive_scenarios)

        print(f"Loading {len(negative_scenarios)} negative scenarios...")
        self._load_negative_scenarios(negative_scenarios)

        print("\nLoading complete!")
        print(f"  Total positive scenarios: {self.positive_collection.count()}")
        print(f"  Total negative scenarios: {self.negative_collection.count()}")

    def _load_positive_scenarios(self, scenarios: list[dict]) -> None:
        """Load positive scenarios into ChromaDB."""
        if not scenarios:
            print("  No positive scenarios to load")
            return

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for idx, scenario in enumerate(scenarios):
            # Use user_query as the document (for embedding)
            documents.append(scenario.get("user_query", ""))

            # Store all other data as metadata
            metadata = {
                "api_name": scenario.get("api_name", ""),
                "function_name": scenario.get("function_name", ""),
                "target_parameter": scenario.get("target_parameter", ""),
                "policy_category": scenario.get("policy_category", "happy_path"),
                "parameters": json.dumps(scenario.get("parameters", {})),
                "rationale": scenario.get("rationale", ""),
            }
            metadatas.append(metadata)

            # Generate unique ID
            scenario_id = f"pos_{scenario.get('api_name', '')}_{scenario.get('function_name', '')}_{idx}"
            ids.append(scenario_id)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            self.positive_collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )

        print(f"  Loaded {len(documents)} positive scenarios")

    def _load_negative_scenarios(self, scenarios: list[dict]) -> None:
        """Load negative scenarios into ChromaDB."""
        if not scenarios:
            print("  No negative scenarios to load")
            return

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for idx, scenario in enumerate(scenarios):
            # Use user_query as the document (for embedding)
            documents.append(scenario.get("user_query", ""))

            # Store all other data as metadata
            metadata = {
                "api_name": scenario.get("api_name", ""),
                "function_name": scenario.get("function_name", ""),
                "target_parameter": scenario.get("target_parameter", ""),
                "policy_violated": scenario.get("policy_violated", ""),
                "violation_type": scenario.get("violation_type", ""),
                "rationale": scenario.get("rationale", ""),
            }
            metadatas.append(metadata)

            # Generate unique ID
            scenario_id = f"neg_{scenario.get('api_name', '')}_{scenario.get('function_name', '')}_{idx}"
            ids.append(scenario_id)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            self.negative_collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )

        print(f"  Loaded {len(documents)} negative scenarios")

    def load_from_directory(self, scenarios_dir: str | Path) -> None:
        """
        Load all scenario files from a directory.

        Args:
            scenarios_dir: Directory containing scenario JSON files
        """
        scenarios_dir = Path(scenarios_dir)

        all_positive = []
        all_negative = []

        # Load all positive scenario files
        for file_path in scenarios_dir.glob("*_positive_scenarios.json"):
            with open(file_path, 'r') as f:
                scenarios = json.load(f)
                all_positive.extend(scenarios)

        # Load all negative scenario files
        for file_path in scenarios_dir.glob("*_negative_scenarios.json"):
            with open(file_path, 'r') as f:
                scenarios = json.load(f)
                all_negative.extend(scenarios)

        # Load into ChromaDB
        self.load_scenarios(all_positive, all_negative)

    def query_positive(self, query_text: str, n_results: int = 5) -> dict:
        """
        Query the positive scenarios collection.

        Args:
            query_text: Query text to search for
            n_results: Number of results to return

        Returns:
            Query results with documents, metadatas, and distances
        """
        return self.positive_collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

    def query_negative(self, query_text: str, n_results: int = 5) -> dict:
        """
        Query the negative scenarios collection.

        Args:
            query_text: Query text to search for
            n_results: Number of results to return

        Returns:
            Query results with documents, metadatas, and distances
        """
        return self.negative_collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

    def query_by_function(
        self, function_name: str, collection: str = "both", n_results: int = 10
    ) -> dict:
        """
        Query scenarios for a specific function.

        Args:
            function_name: Name of the function to query
            collection: "positive", "negative", or "both"
            n_results: Number of results per collection

        Returns:
            Query results
        """
        results = {}

        if collection in ["positive", "both"]:
            pos_results = self.positive_collection.get(
                where={"function_name": function_name},
                limit=n_results
            )
            results["positive"] = pos_results

        if collection in ["negative", "both"]:
            neg_results = self.negative_collection.get(
                where={"function_name": function_name},
                limit=n_results
            )
            results["negative"] = neg_results

        return results

    def get_stats(self) -> dict:
        """Get statistics about the loaded scenarios."""
        stats = {
            "positive_count": self.positive_collection.count(),
            "negative_count": self.negative_collection.count(),
            "total_count": self.positive_collection.count() + self.negative_collection.count(),
        }

        # Get unique APIs in each collection
        try:
            pos_results = self.positive_collection.get()
            neg_results = self.negative_collection.get()

            if pos_results and "metadatas" in pos_results:
                pos_apis = set(meta.get("api_name", "") for meta in pos_results["metadatas"])
                stats["positive_apis"] = sorted(pos_apis)

            if neg_results and "metadatas" in neg_results:
                neg_apis = set(meta.get("api_name", "") for meta in neg_results["metadatas"])
                stats["negative_apis"] = sorted(neg_apis)

        except Exception as e:
            print(f"Error getting stats: {e}")

        return stats

    def reset_collections(self) -> None:
        """Delete and recreate collections with OpenAI embeddings (WARNING: destructive)."""
        print("WARNING: Deleting all collections...")

        self.client.delete_collection("positive_scenarios")
        self.client.delete_collection("negative_scenarios")

        self.positive_collection = self.client.get_or_create_collection(
            name="positive_scenarios",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Policy-compliant (happy path) scenarios",
                "embedding_model": "text-embedding-3-large"
            }
        )

        self.negative_collection = self.client.get_or_create_collection(
            name="negative_scenarios",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Policy-violating (adversarial) scenarios",
                "embedding_model": "text-embedding-3-large"
            }
        )

        print("Collections reset successfully with OpenAI embeddings")


def main():
    """Example usage of ChromaDB loader."""
    from pathlib import Path

    scenarios_dir = Path(__file__).parent / "scenarios"

    # Initialize loader
    loader = ChromaDBLoader()

    # Option 1: Load from directory
    if scenarios_dir.exists():
        loader.load_from_directory(scenarios_dir)
    else:
        print(f"Scenarios directory not found: {scenarios_dir}")

    # Show stats
    stats = loader.get_stats()
    print("\nDatabase Statistics:")
    print(f"  Total scenarios: {stats['total_count']}")
    print(f"  Positive: {stats['positive_count']}")
    print(f"  Negative: {stats['negative_count']}")

    # Example query
    print("\nExample query: 'Set cruise control to 65 mph'")
    results = loader.query_positive("Set cruise control to 65 mph", n_results=3)
    print(f"  Found {len(results['documents'][0])} results")


if __name__ == "__main__":
    main()
