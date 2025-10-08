import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, List, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import Field
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker
from pymilvus.exceptions import MilvusException

# Default timeout for all Milvus operations (in seconds)
DEFAULT_MILVUS_TIMEOUT = 30.0


class EmbeddingService:
    """Service for generating embeddings from text using an external API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding service.

        Args:
            api_url: URL of the embedding API endpoint
            model: Name of the embedding model to use
            api_key: Optional API key for authentication
        """
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.enabled = api_url is not None

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector from text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If embedding service is not configured or API call fails
        """
        if not self.enabled:
            raise ValueError(
                "Embedding service not configured. Set EMBEDDING_API_URL environment variable."
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                payload = {"model": self.model, "input": text}

                response = await client.post(
                    self.api_url, json=payload, headers=headers
                )
                response.raise_for_status()

                data = response.json()

                # Handle OpenAI-compatible response format
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                # Handle direct embedding array response
                elif "embedding" in data:
                    return data["embedding"]
                else:
                    raise ValueError(f"Unexpected API response format: {data}")

        except httpx.HTTPError as e:
            raise ValueError(f"Embedding API request failed: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse embedding API response: {str(e)}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If embedding service is not configured or API call fails
        """
        if not self.enabled:
            raise ValueError(
                "Embedding service not configured. Set EMBEDDING_API_URL environment variable."
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                payload = {"model": self.model, "input": texts}

                response = await client.post(
                    self.api_url, json=payload, headers=headers
                )
                response.raise_for_status()

                data = response.json()

                # Handle OpenAI-compatible response format
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                # Handle direct embedding array response
                elif "embeddings" in data:
                    return data["embeddings"]
                else:
                    raise ValueError(f"Unexpected API response format: {data}")

        except httpx.HTTPError as e:
            raise ValueError(f"Embedding API request failed: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse embedding API response: {str(e)}")


class MilvusConnector:
    def __init__(
        self, uri: str, token: Optional[str] = None, db_name: Optional[str] = "legal"
    ):
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self._client = None
        self._schema_cache: dict[str, dict[str, Any]] = {}

    @property
    def client(self) -> MilvusClient:
        """Lazy initialization of Milvus client."""
        if self._client is None:
            self._client = MilvusClient(
                uri=self.uri, token=self.token, db_name=self.db_name
            )
        return self._client

    async def list_collections(self) -> list[str]:
        """List all collections in the database."""
        try:
            return await asyncio.to_thread(self.client.list_collections)
        except MilvusException as error:
            raise self._to_value_error(error)
        except Exception as error:
            raise ValueError(f"Failed to list collections: {str(error)}") from error

    async def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a collection."""
        try:
            return await asyncio.to_thread(
                self.client.describe_collection, collection_name
            )
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Failed to get collection info: {str(error)}") from error

    async def search_collection(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        drop_ratio: float = 0.2,
        text_field: str = "text",
    ) -> list[dict]:
        """
        Perform full text search on a collection.

        Args:
            collection_name: Name of collection to search
            query_text: Text to search for
            limit: Maximum number of results
            output_fields: Fields to return in results
            drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
            text_field: Field name containing textual content
        """
        try:
            await self._ensure_collection_exists(collection_name)
            resolved_text_field = await self._resolve_text_field(
                collection_name, text_field
            )
            search_params = {"params": {"drop_ratio_search": drop_ratio}}
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=[query_text],
                anns_field=resolved_text_field,
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
            )
            return results
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Search failed: {str(error)}") from error

    async def query_collection(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Query collection using filter expressions."""
        try:
            await self._ensure_collection_exists(collection_name)
            return await asyncio.to_thread(
                self.client.query,
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
            )
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Query failed: {str(error)}") from error

    async def vector_search(
        self,
        collection_name: str,
        vector: list[float],
        vector_field: str = "embedding",
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform vector similarity search on a collection.

        Args:
            collection_name: Name of collection to search
            vector: Query vector
            vector_field: Field containing vectors to search
            limit: Maximum number of results
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
        """
        try:
            await self._ensure_collection_exists(collection_name)
            resolved_vector_field = await self._resolve_vector_field(
                collection_name, vector_field
            )
            search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=[vector],
                anns_field=resolved_vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr,
            )
            return results
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Vector search failed: {str(error)}") from error

    async def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        text_field: str = "text",
        vector: list[float] | None = None,
        vector_field: str = "embedding",
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform hybrid search combining BM25 text search and vector search with RRF ranking.

        Args:
            collection_name: Name of collection to search
            query_text: Text query for BM25 search
            text_field: Field name for text search
            vector: Query vector for dense vector search
            vector_field: Field name for vector search
            limit: Maximum number of results
            output_fields: Fields to return in results
            filter_expr: Optional filter expression
        """
        try:
            await self._ensure_collection_exists(collection_name)
            resolved_text_field = await self._resolve_text_field(
                collection_name, text_field
            )
            resolved_vector_field = await self._resolve_vector_field(
                collection_name, vector_field
            )
            sparse_params = {"params": {"drop_ratio_search": 0.2}}
            dense_params = {"params": {"nprobe": 10}}
            # BM25 search request
            sparse_request = AnnSearchRequest(
                data=[query_text],
                anns_field=resolved_text_field,
                param=sparse_params,
                limit=limit,
            )
            # dense vector search request
            dense_request = AnnSearchRequest(
                data=[vector],
                anns_field=resolved_vector_field,
                param=dense_params,
                limit=limit,
            )
            # hybrid search
            results = await asyncio.to_thread(
                self.client.hybrid_search,
                collection_name=collection_name,
                reqs=[sparse_request, dense_request],
                ranker=RRFRanker(60),
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr,
            )

            return results

        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Hybrid search failed: {str(error)}") from error

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Name of collection
        """
        try:
            await self._ensure_collection_exists(collection_name)
            return await asyncio.to_thread(
                self.client.get_collection_stats, collection_name
            )
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Failed to get collection stats: {str(error)}") from error

    async def multi_vector_search(
        self,
        collection_name: str,
        vectors: list[list[float]],
        vector_field: str = "embedding",
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None,
        search_params: Optional[dict[str, Any]] = None,
    ) -> list[list[dict]]:
        """
        Perform vector similarity search with multiple query vectors.

        Args:
            collection_name: Name of collection to search
            vectors: List of query vectors
            vector_field: Field containing vectors to search
            limit: Maximum number of results per query
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
            search_params: Additional search parameters
        """
        try:
            await self._ensure_collection_exists(collection_name)
            if search_params is None:
                search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
            resolved_vector_field = await self._resolve_vector_field(
                collection_name, vector_field
            )
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=vectors,
                anns_field=resolved_vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr,
            )
            return results
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Multi-vector search failed: {str(error)}") from error

    async def load_collection(
        self, collection_name: str, replica_number: int = 1
    ) -> bool:
        """
        Load a collection into memory for search and query.

        Args:
            collection_name: Name of collection to load
            replica_number: Number of replicas
        """
        try:
            await self._ensure_collection_exists(collection_name)
            await asyncio.to_thread(
                self.client.load_collection,
                collection_name=collection_name,
                replica_number=replica_number,
            )
            return True
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Failed to load collection: {str(error)}") from error

    async def release_collection(self, collection_name: str) -> bool:
        """
        Release a collection from memory.

        Args:
            collection_name: Name of collection to release
        """
        try:
            await self._ensure_collection_exists(collection_name)
            await asyncio.to_thread(
                self.client.release_collection, collection_name=collection_name
            )
            return True
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Failed to release collection: {str(error)}") from error

    async def get_query_segment_info(self, collection_name: str) -> dict[str, Any]:
        """
        Get information about query segments.

        Args:
            collection_name: Name of collection
        """
        try:
            await self._ensure_collection_exists(collection_name)
            return await asyncio.to_thread(
                self.client.get_query_segment_info, collection_name
            )
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(
                f"Failed to get query segment info: {str(error)}"
            ) from error

    async def get_index_info(
        self, collection_name: str, field_name: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Get information about indexes in a collection.

        Args:
            collection_name: Name of collection
            field_name: Optional specific field to get index info for
        """
        try:
            await self._ensure_collection_exists(collection_name)
            return await asyncio.to_thread(
                self.client.describe_index,
                collection_name=collection_name,
                index_name=field_name,
            )
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Failed to get index info: {str(error)}") from error

    async def get_collection_loading_progress(
        self, collection_name: str
    ) -> dict[str, Any]:
        """
        Get the loading progress of a collection.

        Args:
            collection_name: Name of collection
        """
        try:
            await self._ensure_collection_exists(collection_name)
            return await asyncio.to_thread(self.client.get_load_state, collection_name)
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(f"Failed to get loading progress: {str(error)}") from error

    async def list_databases(self) -> list[str]:
        """List all databases in the Milvus instance."""
        try:
            return await asyncio.to_thread(self.client.list_databases)
        except MilvusException as error:
            raise self._to_value_error(error)
        except Exception as error:
            raise ValueError(f"Failed to list databases: {str(error)}") from error

    async def use_database(self, db_name: str) -> bool:
        """Switch to a different database.

        Args:
            db_name: Name of the database to use
        """
        try:
            # Update db_name and reset client for lazy re-initialization
            if self._client is not None and hasattr(self._client, "close"):
                await asyncio.to_thread(self._client.close)
            self.db_name = db_name
            self._client = None  # Reset client to reconnect with new database
            self._schema_cache.clear()
            return True
        except MilvusException as error:
            raise self._to_value_error(error)
        except Exception as error:
            raise ValueError(f"Failed to switch database: {str(error)}") from error

    async def _get_collection_schema(self, collection_name: str) -> dict[str, Any]:
        """Retrieve and cache the collection schema."""
        if collection_name not in self._schema_cache:
            try:
                schema = await asyncio.to_thread(
                    self.client.describe_collection, collection_name
                )
            except MilvusException as error:
                raise self._to_value_error(error, collection_name)
            except Exception as error:
                raise ValueError(f"Failed to retrieve schema: {str(error)}") from error
            self._schema_cache[collection_name] = schema
        return self._schema_cache[collection_name]

    async def _resolve_vector_field(
        self, collection_name: str, requested_field: str
    ) -> str:
        """Resolve the appropriate vector field for a collection."""
        schema = await self._get_collection_schema(collection_name)
        field_name = self._choose_field(
            schema,
            requested_field,
            candidates=[requested_field, "embedding", "vector", "dense_vector"],
            expected_type=DataType.FLOAT_VECTOR,
        )
        if field_name is None:
            raise ValueError(
                f"No FLOAT_VECTOR field available in collection '{collection_name}'."
            )
        return field_name

    async def _resolve_text_field(
        self, collection_name: str, requested_field: str
    ) -> str:
        """Resolve the appropriate sparse vector field for BM25 text search."""
        schema = await self._get_collection_schema(collection_name)
        field_name = self._choose_field(
            schema,
            requested_field,
            candidates=[requested_field, "sparse", "sparse_vector", "text_sparse"],
            expected_type=DataType.SPARSE_FLOAT_VECTOR,
        )
        if field_name is None:
            raise ValueError(
                f"No SPARSE_FLOAT_VECTOR field available for text search in collection '{collection_name}'."
            )
        return field_name

    @staticmethod
    def _choose_field(
        schema: dict[str, Any],
        requested_field: str,
        candidates: list[str],
        expected_type: DataType,
    ) -> Optional[str]:
        """Select the first matching field from candidates that exists in the schema."""

        # Schema from describe_collection has fields directly, not under schema.fields
        fields = schema.get("fields", [])

        def matches(field: dict[str, Any], name: str) -> bool:
            if field.get("name") != name:
                return False
            # The type field contains the integer enum value
            field_type = field.get("type")
            # Compare integer type values
            if isinstance(field_type, int):
                return field_type == expected_type.value
            elif isinstance(field_type, DataType):
                return field_type == expected_type
            return False

        # Ensure requested field is evaluated first
        seen = set()
        ordered_candidates = []
        for candidate in candidates:
            if candidate not in seen and candidate is not None:
                ordered_candidates.append(candidate)
                seen.add(candidate)

        for candidate in ordered_candidates:
            for field in fields:
                if matches(field, candidate):
                    return candidate

        # Fall back to any field of the expected type if specific names not found
        for field in fields:
            field_type = field.get("type")
            # Compare integer type values
            if isinstance(field_type, int) and field_type == expected_type.value:
                return field.get("name")
            elif isinstance(field_type, DataType) and field_type == expected_type:
                return field.get("name")

        return None

    async def _ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure the target collection exists before running an operation."""

        try:
            exists = await asyncio.to_thread(
                self.client.has_collection, collection_name
            )
        except MilvusException as error:
            raise self._to_value_error(error, collection_name)
        except Exception as error:
            raise ValueError(
                f"Failed to verify collection existence: {str(error)}"
            ) from error

        if not exists:
            raise ValueError(f"collection_not_found: '{collection_name}'")

    @staticmethod
    def _is_collection_missing(error: MilvusException) -> bool:
        """Check if the provided error indicates a missing collection."""

        message = str(error).lower()
        code = getattr(error, "code", None)
        return code == 1100 or "not found" in message or "does not exist" in message

    def _to_value_error(
        self, error: Exception, collection_name: Optional[str] = None
    ) -> ValueError:
        """Convert low-level exceptions into normalized ValueError instances."""

        if isinstance(error, MilvusException):
            if self._is_collection_missing(error):
                if collection_name is not None:
                    return ValueError(f"collection_not_found: '{collection_name}'")
                return ValueError("collection_not_found")
            return ValueError(f"Milvus error: {str(error)}")
        return ValueError(str(error))


class MilvusContext:
    def __init__(
        self,
        connector: MilvusConnector,
        embedding_service: EmbeddingService,
        timeout: float = DEFAULT_MILVUS_TIMEOUT,
    ):
        self.connector = connector
        self.embedding_service = embedding_service
        self.timeout = timeout


async def with_timeout(coro, timeout: float, operation_name: str):
    """
    Wrap a coroutine with a timeout.

    Args:
        coro: The coroutine to execute
        timeout: Timeout in seconds
        operation_name: Name of the operation for error messages

    Raises:
        TimeoutError: If the operation exceeds the timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Milvus operation '{operation_name}' timed out after {timeout} seconds"
        )


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[MilvusContext]:
    """Manage application lifecycle for Milvus connector."""
    config = server.config

    connector = MilvusConnector(
        uri=config.get("milvus_uri", "http://localhost:19530"),
        token=config.get("milvus_token"),
        db_name=config.get("db_name", "legal"),
    )

    embedding_service = EmbeddingService(
        api_url=config.get("embedding_api_url"),
        model=config.get("embedding_model", "text-embedding-ada-002"),
        api_key=config.get("embedding_api_key"),
    )

    timeout = config.get("milvus_timeout", DEFAULT_MILVUS_TIMEOUT)

    try:
        yield MilvusContext(connector, embedding_service, timeout)
    finally:
        pass


mcp = FastMCP(name="Milvus", lifespan=server_lifespan)


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_text_search(
    collection_name: Annotated[
        str, Field(description="Name of the collection to search")
    ],
    query_text: Annotated[
        str, Field(description="Text query for BM25 full-text search")
    ],
    ctx: Context,
    limit: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=100)
    ] = 5,
    output_fields: Annotated[
        list[str] | None, Field(description="Fields to include in results")
    ] = None,
    drop_ratio: Annotated[
        float,
        Field(
            description="Proportion of low-frequency terms to ignore (0.0-1.0)",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.2,
    text_field: Annotated[
        str, Field(description="Name of the sparse vector field for BM25 search")
    ] = "sparse",
) -> str:
    """
    Search for documents using full text search (BM25) in a Milvus collection.

    Args:
        collection_name: Name of the collection to search
        query_text: Text to search for
        ctx: FastMCP context
        limit: Maximum number of results to return
        output_fields: Fields to include in results
        drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
        text_field: Name of the sparse vector field for BM25 search
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector

    results = await with_timeout(
        connector.search_collection(
            collection_name=collection_name,
            query_text=query_text,
            limit=limit,
            output_fields=output_fields,
            drop_ratio=drop_ratio,
            text_field=text_field,
        ),
        context.timeout,
        "text_search",
    )

    output = f"Search results for '{query_text}' in collection '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_semantic_search(
    collection_name: Annotated[
        str, Field(description="Name of the collection to search")
    ],
    query_text: Annotated[
        str, Field(description="Natural language query for semantic search")
    ],
    ctx: Context,
    vector_field: Annotated[
        str, Field(description="Field containing vectors to search")
    ] = "embedding",
    limit: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=100)
    ] = 5,
    output_fields: Annotated[
        list[str] | None, Field(description="Fields to include in results")
    ] = None,
    metric_type: Annotated[
        str, Field(description="Distance metric: COSINE, L2, or IP")
    ] = "COSINE",
    filter_expr: Annotated[
        str | None, Field(description="Optional filter expression")
    ] = None,
) -> str:
    """
    Perform semantic search using automatic text embedding.
    This is the recommended tool for natural language queries.

    Args:
        collection_name: Name of the collection to search
        query_text: Natural language query to search for
        ctx: FastMCP context
        vector_field: Field containing vectors to search (default: "embedding")
        limit: Maximum number of results
        output_fields: Fields to include in results
        metric_type: Distance metric (COSINE, L2, IP)
        filter_expr: Optional filter expression
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector
    embedding_service = context.embedding_service

    try:
        # Automatically generate embedding from query text
        search_vector = await with_timeout(
            embedding_service.embed_text(query_text),
            context.timeout,
            "embed_text",
        )
    except ValueError as e:
        return f"Error: Embedding service not available. {str(e)}\nPlease configure EMBEDDING_API_URL, EMBEDDING_MODEL, and optionally EMBEDDING_API_KEY environment variables."
    except TimeoutError as e:
        return f"Error: {str(e)}"

    results = await with_timeout(
        connector.vector_search(
            collection_name=collection_name,
            vector=search_vector,
            vector_field=vector_field,
            limit=limit,
            output_fields=output_fields,
            metric_type=metric_type,
            filter_expr=filter_expr,
        ),
        context.timeout,
        "semantic_search",
    )

    output = f"Semantic search results for '{query_text}' in '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_list_collections(ctx: Context) -> str:
    """List all collections in the current Milvus database."""
    context = ctx.request_context.lifespan_context
    connector = context.connector

    collections = await with_timeout(
        connector.list_collections(),
        context.timeout,
        "list_collections",
    )
    return f"Collections in database:\n{', '.join(collections)}"


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_query(
    collection_name: Annotated[
        str, Field(description="Name of the collection to query")
    ],
    filter_expr: Annotated[
        str,
        Field(description="Filter expression (e.g. 'age > 20' or 'id in [1, 2, 3]')"),
    ],
    ctx: Context,
    output_fields: Annotated[
        list[str] | None, Field(description="Fields to include in results")
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum number of results", ge=1, le=1000)
    ] = 10,
) -> str:
    """
    Query collection using filter expressions to retrieve records matching specific criteria.

    Args:
        collection_name: Name of the collection to query
        filter_expr: Filter expression (e.g. 'age > 20')
        output_fields: Fields to include in results
        limit: Maximum number of results
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector

    results = await with_timeout(
        connector.query_collection(
            collection_name=collection_name,
            filter_expr=filter_expr,
            output_fields=output_fields,
            limit=limit,
        ),
        context.timeout,
        "query",
    )

    output = f"Query results for '{filter_expr}' in collection '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_vector_search(
    collection_name: Annotated[
        str, Field(description="Name of the collection to search")
    ],
    ctx: Context,
    vector_field: Annotated[
        str, Field(description="Field containing vectors to search")
    ] = "embedding",
    limit: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=100)
    ] = 5,
    output_fields: Annotated[
        list[str] | None, Field(description="Fields to include in results")
    ] = None,
    metric_type: Annotated[
        str, Field(description="Distance metric: COSINE, L2, or IP")
    ] = "COSINE",
    filter_expr: Annotated[
        str | None, Field(description="Optional filter expression")
    ] = None,
    vector: Annotated[
        list[float] | None, Field(description="Pre-computed query vector")
    ] = None,
    query_text: Annotated[
        str | None, Field(description="Text to automatically embed as query vector")
    ] = None,
) -> str:
    """
    Perform vector similarity search on a collection.
    Provide either 'vector' directly or 'query_text' for automatic embedding.

    Args:
        collection_name: Name of the collection to search
        vector_field: Field containing vectors to search
        limit: Maximum number of results
        output_fields: Fields to include in results
        metric_type: Distance metric (COSINE, L2, IP)
        filter_expr: Optional filter expression
        vector: Pre-computed query vector (optional if query_text provided)
        query_text: Text to automatically embed (optional if vector provided)
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector
    embedding_service = context.embedding_service

    # Determine which vector to use
    search_vector = vector
    if search_vector is None and query_text is not None:
        # Automatically generate embedding from query text
        try:
            search_vector = await with_timeout(
                embedding_service.embed_text(query_text),
                context.timeout,
                "embed_text",
            )
        except ValueError as e:
            return f"Error generating embedding: {str(e)}\nPlease provide a 'vector' parameter directly or configure the embedding service."
        except TimeoutError as e:
            return f"Error: {str(e)}"

    if search_vector is None:
        return (
            "Error: Either 'vector' or 'query_text' must be provided for vector search."
        )

    results = await with_timeout(
        connector.vector_search(
            collection_name=collection_name,
            vector=search_vector,
            vector_field=vector_field,
            limit=limit,
            output_fields=output_fields,
            metric_type=metric_type,
            filter_expr=filter_expr,
        ),
        context.timeout,
        "vector_search",
    )

    query_desc = f"text '{query_text}'" if query_text else "provided vector"
    output = f"Vector search results for {query_desc} in '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_hybrid_search(
    collection_name: Annotated[str, Field(description="Name of collection to search")],
    query_text: Annotated[
        str, Field(description="Text query for both BM25 and semantic search")
    ],
    ctx: Context,
    text_field: Annotated[
        str, Field(description="Name of the sparse vector field for BM25 search")
    ] = "sparse",
    vector_field: Annotated[
        str, Field(description="Field containing dense vectors for semantic search")
    ] = "embedding",
    limit: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=100)
    ] = 5,
    output_fields: Annotated[
        list[str] | None, Field(description="Fields to include in results")
    ] = None,
    filter_expr: Annotated[
        str | None, Field(description="Optional filter expression")
    ] = None,
    vector: Annotated[
        list[float] | None,
        Field(
            description="Optional pre-computed query vector (auto-generated if not provided)"
        ),
    ] = None,
) -> str:
    """
    Perform hybrid search combining BM25 text search and vector search with RRF ranking.
    If only query_text is provided, it will be used for both text search and automatic embedding.

    Args:
        collection_name: Name of collection to search
        query_text: Text query for both BM25 and semantic search
        ctx: FastMCP context
        text_field: Name of the sparse vector field for BM25 search
        vector_field: Field name for dense vector search
        limit: Maximum number of results
        output_fields: Fields to return in results
        filter_expr: Optional filter expression
        vector: Query vector for dense vector search (optional, auto-generated if not provided)
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector
    embedding_service = context.embedding_service

    # Determine which vector to use
    search_vector = vector
    if search_vector is None:
        # Automatically generate embedding from query text
        try:
            search_vector = await with_timeout(
                embedding_service.embed_text(query_text),
                context.timeout,
                "embed_text",
            )
        except ValueError as e:
            return f"Error generating embedding: {str(e)}\nPlease provide a 'vector' parameter directly or configure the embedding service."
        except TimeoutError as e:
            return f"Error: {str(e)}"

    results = await with_timeout(
        connector.hybrid_search(
            collection_name=collection_name,
            query_text=query_text,
            text_field=text_field,
            vector=search_vector,
            vector_field=vector_field,
            limit=limit,
            output_fields=output_fields,
            filter_expr=filter_expr,
        ),
        context.timeout,
        "hybrid_search",
    )

    output = (
        f"Hybrid search results for text '{query_text}' in '{collection_name}':\n\n"
    )
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool
async def milvus_load_collection(
    collection_name: Annotated[
        str, Field(description="Name of collection to load into memory")
    ],
    ctx: Context,
    replica_number: Annotated[
        int, Field(description="Number of replicas to load", ge=1)
    ] = 1,
) -> str:
    """
    Load a collection into memory for search and query operations.

    Args:
        collection_name: Name of collection to load
        ctx: FastMCP context
        replica_number: Number of replicas
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector

    await with_timeout(
        connector.load_collection(
            collection_name=collection_name, replica_number=replica_number
        ),
        context.timeout,
        "load_collection",
    )

    return f"Collection '{collection_name}' loaded successfully with {replica_number} replica(s)"


@mcp.tool
async def milvus_release_collection(
    collection_name: Annotated[
        str, Field(description="Name of collection to release from memory")
    ],
    ctx: Context,
) -> str:
    """
    Release a collection from memory to free up resources.

    Args:
        collection_name: Name of collection to release
        ctx: FastMCP context
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector

    await with_timeout(
        connector.release_collection(collection_name=collection_name),
        context.timeout,
        "release_collection",
    )

    return f"Collection '{collection_name}' released successfully"


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_list_databases(ctx: Context) -> str:
    """List all databases in the Milvus instance."""
    context = ctx.request_context.lifespan_context
    connector = context.connector

    databases = await with_timeout(
        connector.list_databases(),
        context.timeout,
        "list_databases",
    )
    return f"Databases in Milvus instance:\n{', '.join(databases)}"


@mcp.tool
async def milvus_use_database(
    db_name: Annotated[str, Field(description="Name of the database to switch to")],
    ctx: Context,
) -> str:
    """
    Switch to a different database context.

    Args:
        db_name: Name of the database to use
        ctx: FastMCP context
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector

    await with_timeout(
        connector.use_database(db_name),
        context.timeout,
        "use_database",
    )

    return f"Switched to database '{db_name}' successfully"


@mcp.tool(annotations={"readOnlyHint": True})
async def milvus_get_collection_info(
    collection_name: Annotated[
        str, Field(description="Name of collection to get information about")
    ],
    ctx: Context,
) -> str:
    """
    Get detailed schema and metadata information about a specific collection.

    Args:
        collection_name: Name of collection to inspect
        ctx: FastMCP context
    """
    context = ctx.request_context.lifespan_context
    connector = context.connector

    collection_info = await with_timeout(
        connector.get_collection_info(collection_name),
        context.timeout,
        "get_collection_info",
    )
    return f"Collection information:\n{collection_info}"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Milvus MCP Server")
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default="http://localhost:19530",
        help="Milvus server URI",
    )
    parser.add_argument(
        "--milvus-token", type=str, default=None, help="Milvus authentication token"
    )
    parser.add_argument(
        "--milvus-db", type=str, default="legal", help="Milvus database name"
    )
    parser.add_argument(
        "--milvus-timeout",
        type=float,
        default=DEFAULT_MILVUS_TIMEOUT,
        help=f"Timeout for Milvus operations in seconds (default: {DEFAULT_MILVUS_TIMEOUT})",
    )
    parser.add_argument(
        "--embedding-api-url",
        type=str,
        default=None,
        help="Embedding API URL (e.g., http://localhost:8081/v1/embeddings)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-ada-002",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-api-key", type=str, default=None, help="Embedding API key"
    )
    parser.add_argument("--sse", action="store_true", help="Enable SSE mode")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number for SSE server"
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_arguments()
    mcp.config = {
        "milvus_uri": os.environ.get("MILVUS_URI", args.milvus_uri),
        "milvus_token": os.environ.get("MILVUS_TOKEN", args.milvus_token),
        "db_name": os.environ.get("MILVUS_DB", args.milvus_db),
        "milvus_timeout": float(os.environ.get("MILVUS_TIMEOUT", args.milvus_timeout)),
        "embedding_api_url": os.environ.get(
            "EMBEDDING_API_URL", args.embedding_api_url
        ),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", args.embedding_model),
        "embedding_api_key": os.environ.get(
            "EMBEDDING_API_KEY", args.embedding_api_key
        ),
    }
    if args.sse:
        mcp.run(transport="sse", port=args.port, host="0.0.0.0")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
