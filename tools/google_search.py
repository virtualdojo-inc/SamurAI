"""Tool for performing Google web searches via Vertex AI grounding."""

import os

from google import genai
from google.genai.types import Tool as GenaiTool, GoogleSearch, GenerateContentConfig
from langchain_core.tools import tool


@tool
def google_search(query: str) -> str:
    """Search the web using Google Search.

    IMPORTANT: Only use this tool when the user explicitly asks you to search
    the web, google something, or look something up online. Do NOT use this
    tool proactively or to answer questions you already know the answer to.

    Args:
        query: The search query string.
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GCP_PROJECT_ID"),
        location="global",
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=query,
        config=GenerateContentConfig(
            tools=[GenaiTool(google_search=GoogleSearch())],
        ),
    )

    parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.text:
                parts.append(part.text)

    if not parts:
        return "No search results found."

    result = "\n".join(parts)

    # Append grounding sources if available
    metadata = response.candidates[0].grounding_metadata
    if metadata and metadata.grounding_supports:
        sources = set()
        for support in metadata.grounding_supports:
            if support.grounding_chunk_indices:
                for idx in support.grounding_chunk_indices:
                    chunks = metadata.grounding_chunks
                    if chunks and idx < len(chunks) and chunks[idx].web:
                        sources.add(
                            f"- [{chunks[idx].web.title}]({chunks[idx].web.uri})"
                        )
        if sources:
            result += "\n\nSources:\n" + "\n".join(sorted(sources))

    return result
