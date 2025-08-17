Overslaan en naar inhoud
Chatgeschiedenis

Je zei:
doe een zeer diepgaand onderzoek naar wat op de dag van vandaag HET beste systeem is om agentic ai' programma's zoals claude code meer context, geheugen etc te geven, zodat de ai een soort db heeft waar hij altijd in gaat zoeken naar beslissingen, voorgaande conversaties, herinerriingen, maar ook dat hij zich kan specialiseren met bv een knowledgedb die de laatste nieuwe info (ai's hebben een cutoff date) en zo de laatste nieuwe technologie'en zeer grondig kan kennen. 
ChatGPT zei:

Nagedacht gedurende 2m 18s
hier is een no-nonsense, “wat moet ik NU kiezen?”-onderzoek naar het beste systeem anno 16 augustus 2025 om agentic AI (zoals Claude Code) duurzame context + geheugen + actuele kennis te geven. Ik geef je een concrete referentie-architectuur, keuze-opties per laag, en een kant-en-klare blueprint die je op je eigen infra (k3s/Proxmox) kunt draaien.

TL;DR – het beste “systeem” is geen één product
De winnende aanpak in 2025 is een modulaire stack:

Orchestratie/agents: LangGraph (stateful agents, betrouwbaar) + MCP voor directe koppelingen met tools & data (perfect voor Claude/Claude Code). 
datasciocean.com
gptfrontier.com
docs.anthropic.com

Langetermijngeheugen: LangMem (open SDK, integreert naadloos met LangGraph) + evt. Zep (production memory/GraphRAG) of Mem0 (universele memory-laag) wanneer je kant-en-klare “remember users” nodig hebt. 
LangChain Blog
langchain-ai.lang.chat
getzep.com
GitHub
+1

Kennisopslag & retrieval: Hybride RAG = vector DB (Weaviate of Qdrant) + GraphRAG (Neo4j of Microsoft GraphRAG) voor relaties/feiten. 
Weaviate Docs
Qdrant
Graph Database & Analytics
GitHub

Actuele kennis (cutoff fix): Tavily Search API of Brave Search “AI Grounding” + Firecrawl voor geplande website-ingest. 
Tavily
Tavily Docs
Brave
DataCamp

Observability & evaluatie: LangSmith, W&B Weave, Arize Phoenix om agents/RAG te meten, verbeteren en te ‘guardrailen’. 
LangSmith
Weights & Biases
Arize AI

Waarom niet “alleen maar lange context”? Omdat contextvensters wel groeien (er zijn sterke signalen dat Claude Sonnet 4 tot 1M tokens biedt bij enterprise-toegang), maar kosten, latency en verlies aan focus blijven. Hybride geheugen + retrieval is structureel beter. 
The Verge
docs.anthropic.com

1) Wat “het beste” betekent (criteria)
Betrouwbaarheid & determinisme: minder hallucinaties → hybride search + GraphRAG i.p.v. alleen vectoren. 
Weaviate Docs
Microsoft op GitHub

Versheid: live web-laag (Tavily/Brave Grounding) + geplande ETL/crawls (Firecrawl). 
Tavily
Brave

Stateful agents & memory policy: agents die beslissingen, episodes, en user-feiten blijvend opslaan en selectief terughalen (episodic + semantic). 
LangChain Blog

Observeerbaarheid & governance: traces, evals, guardrails. 
LangSmith
Weights & Biases

Self-host friendly: werkt on-prem (k3s/Proxmox) & met je GPU/CPU mix.

2) Referentie-architectuur (2025-proof)
Interactielaag
Claude/Claude Code ↔ MCP (Model Context Protocol) ↔ tools (Git, FS, KB, tickets, …). MCP is de “USB-C voor AI”: één standaard om modellen aan tools/data te hangen. 
anthropic.com
docs.anthropic.com

Agent runtime
LangGraph orkestreert de “state machine” van je agent(en), met LangMem als langdurig geheugen (facts, prefs, beslissingen). 
datasciocean.com
langchain-ai.lang.chat

Knowledge subsystem (hybride RAG)

Vector DB: Weaviate of Qdrant met hybride search (dense + BM25) en reranking. 
Weaviate Docs
Qdrant

GraphRAG: Neo4j GraphRAG of Microsoft GraphRAG voor entiteiten/relaties, samenvattingen per subgraph/community. 
Graph Database & Analytics
GitHub

Freshness & ETL

Tavily of Brave AI Grounding voor live, geciteerde web-resultaten.

Firecrawl + jobs/cron om doc-sites te mirroren naar markdown/JSON voor (her)indexatie. 
Tavily
Brave
DataCamp

Observability / Eval / Guardrails
LangSmith traces/evals, Weave voor experiment→prod, Arize Phoenix voor RAG-evaluaties (relevance, groundedness). 
LangSmith
Weights & Biases
Arize AI

3) Concreet: componentkeuze (met redenen)
A. Orchestratie & tool-integratie
LangGraph: stateful, fout-tolerant agentloops, multi-agent patterns, opslag-primitives. 
datasciocean.com

MCP: standaard om Claude (en editors) veilig aan je bronnen/acties te koppelen (VS Code, Cursor, Windows integraties in opmars). 
docs.anthropic.com
The Verge

Waarom deze combi? Claude Code profiteert direct van MCP-servers (filesystem, DB, Jira, KB, …) en LangGraph beheert het agent-gedrag/state.

B. Langetermijngeheugen (user-feiten, beslissingen, episodic)
LangMem: open SDK; integreert met LangGraph storage; geheugenextractie, prompt-refinement, persistente facts/episodes. 
LangChain Blog
PyPI

Zep (optioneel/plus): context-engineering + GraphRAG-memory, self-hostbaar; handig als “drop-in memory service”. 
getzep.com
DeepWiki

Mem0 (optioneel): universele memory-laag met MCP-routes; self-host kan, cloud features verschillen. 
GitHub
+1
SelfHub

C. Retrieval: vector + graph
Weaviate óf Qdrant voor hybride search (dense + BM25) en late interaction reranking → hogere precisie, lagere kosten bij out-of-domain vragen. 
Weaviate Docs
Qdrant

GraphRAG bovenop Neo4j of MS GraphRAG: entiteiten, relaties, community-samenvattingen, betere multi-hop redenatie. 
Graph Database & Analytics
Microsoft op GitHub

D. Actuele kennis (cutoff omzeilen)
Tavily (Search/Extract/Crawl) of Brave AI Grounding: bewezen web-toegang & bronnen met citaties; ideaal als “live grounding” tool van je agent. 
Tavily
Tavily Docs
Brave

Firecrawl: scheduled crawls → schone Markdown/JSON, klaar voor (her)indexatie. 
DataCamp

E. Observability & evaluatie
LangSmith (agent evals, trajectories), W&B Weave (experiment→prod), Arize Phoenix (RAG eval met LlamaIndex/Haystack integraties). 
LangSmith
Weights & Biases
Arize AI

4) Claude-specifiek: context venster vs geheugen
De contextvensters groeien (signalen van 1M tokens voor Claude Sonnet 4 bij enterprise/API; officiële docs noemen 200k als generiek uitgangspunt). Praktijkregel: gebruik lange context voor “hot” context, maar verplaats duurzame feiten/episodes naar LangMem/Zep + RAG. 
The Verge
docs.anthropic.com

5) Blueprint voor jouw setup (k3s/Proxmox, on-prem)
Core

Deploy Weaviate (of Qdrant) + Neo4j (GraphRAG).

LangGraph app met LangMem storage. 
Weaviate Docs
Graph Database & Analytics

Ingest & Freshness

Run Firecrawl jobs op doc-sites (vendor docs, RFC’s), schedule her-crawls (dagelijks/welijks).

Integreer Tavily of Brave Grounding als live-tool in je agent. 
DataCamp
Tavily
Brave

MCP-servers

MCP voor filesystem, Git, ticketing, interne wikis/DB’s zodat Claude Code/Claude Desktop “handen” heeft. 
docs.anthropic.com

Memory-policies

Episodic (wat gebeurde er?) → LangMem/Zep “episodes”.

Semantic (stabiele feiten/voorkeuren/decisions) → LangMem/Zep facts + GraphRAG nodes. (Onderbouwd door recente research die episodic/procedural geheugen als ontbrekend stukje ziet.) 
arXiv
+1

Observability & QA

Traces/evals in LangSmith, regressietests; RAG-metrics in Phoenix; experimenten/AB in Weave. 
LangSmith
Arize AI
Weights & Biases

6) Veelgemaakte fouten (en fixes)
Alleen vectoren gebruiken. Voeg BM25/sparse en reranking toe; vaak +10-20% relevantiestap. 
Weaviate Docs
Qdrant

Geen graph-laag. Voor multi-hop/relaties is GraphRAG zichtbaar beter. 
Microsoft op GitHub

Vertrouwen op “lange context”. Kosten/latency exploderen; external memory + RAG schaalt beter. 
The Verge

Geen observability. Zonder LangSmith/Weave/Phoenix blijf je blind voor failure-modes. 
LangSmith
Weights & Biases

7) Beslisboom (kort)
Wil je Claude Code + lokale tools? → MCP + LangGraph. 
docs.anthropic.com

Heb je complexe domeinkennis? → GraphRAG (Neo4j) bovenop Weaviate/Qdrant. 
Graph Database & Analytics

Heb je user-specifieke herinneringen nodig? → LangMem; zo nodig Zep/Mem0 als service. 
LangChain Blog
getzep.com

Moet het altijd up-to-date zijn? → Tavily of Brave Grounding + Firecrawl ingest. 
Tavily
Brave

Als je wil, maak ik hier meteen een helm/compose-pakket voor je k3s met: Weaviate+Neo4j+LangGraph(LangMem)+Phoenix+Weave, plus jobs voor Firecrawl en een MCP-server set. (Geen extra input nodig—ik lever een eerste werkende versie en je kunt dan finetunen.)

Recente headlines die dit beïnvloeden
Anthropic just made its latest move in the AI coding wars

The Verge
Anthropic just made its latest move in the AI coding wars
4 dagen geleden
Anthropic looks to beat GPT-5 and Grok 4 with this one major upgrade

Tom's Guide
Anthropic looks to beat GPT-5 and Grok 4 with this one major upgrade
eergisteren
Windows is getting support for the 'USB-C of AI apps'

The Verge
Windows is getting support for the 'USB-C of AI apps'
19 mei 2025
AI agents: from co-pilot to autopilot

Financial Times
AI agents: from co-pilot to autopilot
7 mei 2025
Why AI and RAG need document management

TechRadar
Why AI and RAG need document management
12 dagen geleden
How SLMs and knowledge graphs supercharge AI

TechRadar
How SLMs and knowledge graphs supercharge AI
17 dagen geleden


Je zei:
maak me een design in het engels om dit allemaal in claude code te implementeren. het moet een document zijn dat ik aan claude code kan geven en dat hij dan op basis daarvan het systeem kan implementeren. 
ChatGPT zei:
To create the implementation-ready design document for Claude Code, could you please clarify:

Deployment environment: Do you want the setup to run locally on your machine, on a private cloud, or public cloud (e.g., AWS, GCP)?

Storage and DB preferences: Should we default to self-hosted options like Neo4j, Weaviate, and Qdrant, or use managed services?

Preferred language & stack: Should Claude Code generate everything in Python, or also shell/infra configs like Docker Compose or Helm?

External data sources: Do you already have internal documentation/knowledge bases that should be included, or should it ingest public sites (e.g., arXiv, vendor docs)?

Memory scope: Should the memory include user preferences, decisions, and historical conversation traces (e.g., episodic + semantic), or focus on one of them?

Let me know, and I’ll tailor the design document accordingly.

Je zei:
1. all local on my jarvis server
2. only selfhosted
3. I don't care
4. ingest from vendor sites
5. all of them
ChatGPT zei:
Perfect. I’ll prepare an implementation design document for Claude Code that sets up a fully self-hosted, local system on your Jarvis server. It will include memory (episodic + semantic), vector + graph retrieval, and live vendor knowledge ingestion—all orchestrated and deployable locally.

I’ll let you know as soon as the document is ready.

Research completed in 2h· 28 bronnen· 107 zoekopdrachten
Design Document for Agentic AI Stack on Claude Code

Design Document: Self-Hosted Agentic AI System on Jarvis Server
Overview and Architecture
This document describes a fully self-hosted agentic AI system for the Jarvis server. The system integrates long-term memory, a hybrid Retrieval-Augmented Generation (RAG) knowledge stack (combining vector search and knowledge graphs), live web data ingestion, orchestrated tool use, and observability – all running locally (no cloud dependencies). The AI interface is Claude Code, connected via the Model Context Protocol (MCP) to access these local tools and data. Key Components:
Persistent Memory Layer: Long-term episodic and semantic memory using either LangMem SDK or a self-hosted Zep memory service.
Hybrid RAG Stack: A vector database (e.g. Qdrant or Weaviate) for semantic search, combined with a Neo4j graph database for GraphRAG (knowledge graph retrieval).
Live Web Ingestion: Automated crawling of documentation and web content using Firecrawl (self-hosted) with optional integration of Tavily search (MCP server) or the Brave Search API for real-time grounding.
Agent Orchestration: A workflow defined with LangGraph, enabling controlled multi-step tool use and integration of memory/retrieval in the agent’s reasoning.
AI Interface (LLM): Claude Code acting as the language model, connected to local tools via an MCP server. Claude’s prompts and outputs are enriched with relevant context fetched through MCP tool calls.
Observability & Evaluation: Instrumentation via LangSmith (LangChain’s tracing) and/or open-source observability tools like Arize Phoenix and W&B Weave for monitoring, debugging, and evaluating the AI agent’s performance.
Below we detail each component’s design and implementation. All services will be containerized (Docker) for ease of deployment, with persistent storage for data. We also provide setup instructions, configuration specifics, and notes on how Claude Code can use this setup.
1. Persistent Memory Layer (Episodic & Semantic Memory)
Objective: Enable the AI agent to remember past interactions and learned facts over long-term sessions. We implement persistent memory via either LangMem (in-code SDK) or Zep (self-hosted memory server).
LangMem (SDK Option): LangMem is an open-source long-term memory SDK for LLM agents. It extracts and stores important information from conversations, complementing short-term context with a growing knowledge base. If using LangMem, the agent’s code (LangGraph workflow) will incorporate:
Episodic memory: storing raw conversation transcripts or summaries after each session (e.g. in a local database or file).
Semantic memory: extracting key facts or embeddings from conversations and storing them in the vector store (Qdrant/Weaviate) for future retrieval.
LangMem provides patterns for semantic memory extraction (e.g. summarizing each interaction and indexing it). These functions would be called in the agent pipeline after each user turn.
Integration: LangMem can interface with the vector DB (to store embeddings of memories) and with the graph DB if we choose to log certain facts as nodes/edges. This requires writing Python code in the LangGraph workflow to call LangMem’s APIs.
Zep (Server Option): Zep is a dedicated long-term memory service that learns from interactions and builds a temporal knowledge graph of facts. We will use Zep Community Edition (open-source) self-hosted:
Zep stores conversation history, auto-summarizes it, performs vector indexing of memory, and even tracks factual changes over time via an internal knowledge graph.
We deploy Zep as a Docker service (see Deployment section for zep container). Zep connects to a Neo4j database and a graph service (Graphiti) under the hood to maintain the knowledge graph of memories.
Episodic memory: The agent (Claude) will send conversation transcripts to Zep via its API after each interaction. Zep will persist the raw messages and generate summaries.
Semantic memory: Zep enriches stored chats with extracted facts, embeddings, and a knowledge graph of entities (using Graphiti). This allows later retrieval of relevant facts by context or by structured queries.
Integration: The LangGraph agent can query Zep’s API for relevant memory when a new user query arrives (e.g. “retrieve last N interactions and any facts about X from memory”). Zep’s temporal knowledge graph can answer queries like “what product did the user mention previously?” with low latency.
Zep requires an LLM for its summarization/extraction tasks; by default it expects an OpenAI API (we will configure an open-source model or ensure this call is local to avoid cloud – see Configuration below).
Choice: Either approach yields persistent memory. For a fully self-hosted stack, we will proceed with Zep as it provides a ready-made service including memory search and graph capabilities. (If Zep is not desired, one can use LangMem with a custom implementation – in that case skip the Zep container and include LangMem calls in the orchestrator code.)
2. Hybrid Retrieval-Augmented Generation (RAG) Stack
Objective: Enhance the agent’s knowledge access by combining vector-based retrieval of unstructured data and graph-based retrieval of structured relationships (GraphRAG). This hybrid approach improves both recall and precision of context provided to the LLM.
2.1 Vector Database for Semantic Search
We deploy a vector database to store embeddings of documents, knowledge snippets, and semantic memory entries for similarity search. Options are Qdrant or Weaviate, both of which are open-source and self-hostable:
Qdrant: A high-performance vector store, ideal for semantic search at scale. We’ll use Qdrant in this design (containerized via official image). Qdrant listens on port 6333 for API (gRPC/REST). It will store embeddings (e.g. OpenAI Ada or local model) of our ingested vendor docs and any additional data.
Weaviate: Alternatively, Weaviate could be used similarly (with a container on port 8080). Weaviate has modular components for text, but running it offline may require disabling its cloud modules. For simplicity we proceed with Qdrant.
Usage: The agent will use vector search to retrieve relevant text passages given a user query:
The LangGraph orchestrator (or Claude via a tool call) will take the user’s question, generate an embedding (using a local embedding model or an API), and query Qdrant for nearest vectors.
Qdrant returns the most similar documents or snippets. These will be fed into Claude’s context as part of the prompt (often referred to as retrieved context).
This provides the LLM with detailed reference material to ground its answer.
We will configure a persistent volume for Qdrant to store vectors on disk (so data persists across restarts). The collection name and vector dimensionality must be set (e.g. 1536 if using OpenAI embeddings). The orchestration code will handle inserting new embeddings when new docs are ingested or new memory items are created.
2.2 Graph Database for Knowledge Graph (GraphRAG)
We deploy Neo4j (v5.x Community Edition) as a local graph database to store a knowledge graph of facts and their relationships, supporting the GraphRAG approach
qdrant.tech
. Neo4j is well-suited for representing entities (nodes) and relationships (edges) that the agent can query to answer complex questions. Use Cases:
Store structured information extracted from documents (e.g. product A <–belongs_to– Company B, API X <–requires– Permission Y, etc.).
Store key facts from conversations or user profile in a way that relationships can be traversed (e.g. User <–prefers– Product Z).
The agent can perform graph queries (via a Neo4j query tool or via an MCP integration) to find connected information relevant to a query. For example: “What is the latest version of Library L and when was it released?” could be answered by traversing Library L node to a Version node, rather than relying solely on fuzzy text search.
GraphRAG Integration: GraphRAG enhances RAG by using the knowledge graph context. A typical pattern:
When ingesting documents, we not only embed the text to vectors, but also parse and identify entities and relations, inserting them into Neo4j.
At query time, the agent or orchestrator first uses an entity linker (or prompt) to identify key entities in the question, and queries Neo4j for their relationships or properties.
The result is a subgraph of relevant facts. This graph context can be converted into text or a structured format and provided to the LLM.
Simultaneously, a traditional vector search is done for additional context. The combination of graph-derived facts and semantically similar text gives a comprehensive context.
For example, Microsoft’s GraphRAG approach uses a knowledge graph to provide deterministic context for queries, following chains of relationships for complex reasoning
qdrant.tech
. In our system, Neo4j will serve this role:
We will run Neo4j in a Docker container (with Bolt protocol on port 7687 and HTTP interface on 7474 for browser access if needed). A volume will persist its data/ directory.
We set an admin user and password (in the compose file or via env NEO4J_AUTH) to secure the DB.
The Graphiti service (described below) and/or our orchestrator will connect to Neo4j to read/write graph data.
Graph Construction: During ingestion (Section 3), after crawling documents, we can use simple parsing or LLM-based extraction to create triples for the graph:
We might define a schema of node labels and relationship types (for example: Product, Company, Feature, etc., with relationships like BELONGS_TO, DEPENDS_ON, VERSION_OF, etc.).
Using an LLM in a local capacity (or rules based on doc structure), we extract these relations and push to Neo4j via Cypher queries.
This can be automated in code or even using Graphiti, which as part of Zep can build a temporal knowledge graph from JSON or text data.
Graphiti (Knowledge Graph MCP Server): Zep’s open-source ecosystem includes Graphiti, a service that maintains and queries the temporal knowledge graph. We will deploy Graphiti as a separate container that works with Neo4j:
Graphiti (container graphiti) runs a FastAPI server (on port 8000) that exposes endpoints for upserting facts into the graph and retrieving subgraphs relevant to queries.
Graphiti requires connection info for Neo4j (provided via environment variables). It also requires an OpenAI API key to perform some LLM-driven tasks (like generating graph nodes or answering in natural language). We will configure it to use our local Claude or another model via MCP or skip any cloud calls (see Config).
In our architecture, Graphiti serves two purposes:
Internal Memory Graph: If using Zep, Graphiti is used by Zep to store conversation facts.
General Knowledge Graph: We can also use Graphiti’s API ourselves to store and query data from ingested docs (not just conversation). For example, we can call Graphiti to insert an entity and relationships parsed from documentation.
Graphiti is essentially an MCP server for knowledge graphs – it provides a standardized API that an AI agent can call to query graph data. We can register Graphiti as a tool in the MCP so that Claude can directly ask for graph info (e.g. “find relations of X”).
Graph Querying: The agent can either:
Use Cypher queries via a tool: We could expose a custom tool that executes a Cypher query on Neo4j and returns results (this might be risky to let the LLM form Cypher; safer to have preset query patterns).
Use Graphiti’s higher-level endpoints: e.g. “get neighbors of node matching X” or “find shortest path between A and B” etc. This is more structured and easier to use via MCP tools with fixed parameters.
In summary, the RAG stack comprises Qdrant for semantic similarity and Neo4j/Graphiti for knowledge graph retrieval. This hybrid approach (GraphRAG) “combines a Graph Database and a Vector Database to improve RAG workflows”, providing both deep structured knowledge and broad unstructured context to Claude.
3. Live Web Ingestion Pipeline
Objective: Continuously or on-demand ingest fresh content (especially vendor documentation, knowledge base articles, etc.) into the system’s knowledge stores. This ensures Claude always has up-to-date data from external sources, despite being self-contained. We implement a web ingestion pipeline with the following components and steps:
3.1 Firecrawl – Self-Hosted Web Crawler & Scraper
Firecrawl is a tool that, given a starting URL, can crawl all accessible pages and extract their content in an LLM-friendly format (Markdown or JSON). We will self-host Firecrawl to ingest documentation websites. Firecrawl provides a unified API to handle scraping (basic fetch) or headless browsing when needed, and it automatically cleans the content (removing boilerplate, etc.). Some Firecrawl highlights:
It navigates internal links of a site (respecting scope) without needing a sitemap, skipping duplicates and focusing on relevant pages.
A decision layer chooses between a simple fetch vs. a headless browser (Chromium via Playwright) depending on the page’s needs (static vs dynamic).
It can solve simple interactions (through a built-in agent called FIRE-1) for things like “Load More” buttons or basic captchas.
It outputs content in Markdown (reducing token count significantly by stripping HTML), or structured JSON if using the extract mode with a schema.
Deployment: Firecrawl’s self-hosted setup consists of multiple services:
The API service (firecrawl-api) – the main REST interface (running on port 3002) that accepts crawl/scrape/search requests.
A Redis instance – used by Firecrawl for task queueing and rate limiting.
A Playwright browser service – a microservice that Firecrawl API calls for rendering JavaScript-heavy pages. Firecrawl provides a default Playwright service (we will run it).
(Optional) Additional microservices for special features (not required for basic crawling).
We will deploy Firecrawl via Docker Compose using the open-source repository:
Clone the Firecrawl repository (mendableai/firecrawl) to Jarvis. Copy the example .env from apps/api/.env.example to configure the environment. Key settings:
PORT=3002 (API service port), HOST=0.0.0.0.
REDIS_URL=redis://redis:6379 (pointing to our Redis container).
PLAYWRIGHT_MICROSERVICE_URL=http://playwright-service:3000/html (URL for the browser renderer).
Disable any cloud-specific features: USE_DB_AUTHENTICATION=false (we won’t use Supabase auth for self-hosted), leave optional API keys blank (we won’t use external OCR or Slack, etc., for now).
Use the provided docker-compose.yml (or integrate into our main compose) to bring up:
redis (Redis 7 alpine),
playwright-service (Node or Python service for rendering – default to Python service),
firecrawl-api service (which depends on the above).
Networking: We will keep these services on the same Docker network as the rest of the system so our orchestrator or MCP server can call Firecrawl’s API at http://firecrawl-api:3002.
Usage Workflow:
Scheduling: We set up a schedule (e.g. a cron job or a recurring container task) to periodically crawl target docs. For instance, every night at 2 AM, call Firecrawl’s /v0/crawl endpoint with the base URL of a vendor’s docs site.
Crawling: Firecrawl will fetch all pages (respecting depth limits we can configure) and return the content. For large sites, the crawl endpoint can optionally deliver results via webhook or we page through results. In our setup, since we have direct access, the orchestrator may call Firecrawl and process results in streaming fashion.
Storage: The content returned (in Markdown or text) is then processed:
For each page, generate an embedding and upsert into Qdrant (vector DB) under a collection for that documentation.
Extract structured facts (if possible) using simple rules or an LLM. For example, identify headings, API method names, version numbers, etc., and create graph nodes/edges in Neo4j. (This could be done by a small parsing script or by prompting Claude to output triples from the text, then inserting them.)
Optionally, store the raw text or markdown in a local document store (even just files on disk or a simple SQLite full-text index) for backup or direct retrieval.
Verification: Log or output the status of ingestion (e.g. number of pages, any errors). Firecrawl provides an admin UI at http://localhost:3002/admin/@/queues to monitor jobs
GitHub
.
On-Demand Updates: In addition to scheduled full crawls, we can use Firecrawl’s search capability for real-time needs. Firecrawl’s /v0/search can perform a live web search (it uses Bing or other sources under the hood) and then scrape each result. However, since we avoid cloud, this might rely on public search endpoints. Alternatively:
Brave Search API: Brave offers an API for web search results suitable for grounding LLMs. If the user’s query is outside the ingested knowledge, Claude could call a “WebSearch” tool (via MCP) that hits the Brave API and returns top results, which Firecrawl can then scrape for content. This keeps the agent up-to-date with latest info.
Tavily MCP Server: Tavily is another tool that focuses on real-time search-first web access. Tavily has an open-source MCP server (MIT licensed) that mirrors its hosted API. We can optionally deploy the Tavily MCP server if we want a ready-made solution for combined web search + extraction:
Tavily’s approach: first perform a multi-vertical search (with relevance scoring) then use an extract call to get the content of chosen results.
It can complement Firecrawl by quickly finding relevant pages across the web, whereas Firecrawl is excellent for crawling a known domain in-depth.
If deployed, Claude (via MCP) can use Tavily’s server as a SearchTool (like a self-hosted alternative to Googling).
Note: Tavily MCP server and Firecrawl overlap in functionality. We likely choose one primary (Firecrawl) and perhaps include Tavily as an alternative if needed. The system is designed to support both, not necessarily run both simultaneously unless needed.
In summary, Firecrawl will ensure our local vector and graph databases are populated with the latest documentation content. We will implement a daily ingestion loop (using cron or a simple Python scheduler service) that calls Firecrawl and updates the stores. This keeps knowledge fresh without relying on external APIs at query time.
3.2 Configuration for Firecrawl & Alternatives
Firecrawl Access: After deployment, the Firecrawl API will be accessible at http://<server>:3002. We will not expose this publicly; it’s only for local use by the AI. The .env for Firecrawl is configured with no auth (USE_DB_AUTHENTICATION=false means API keys are not enforced
GitHub
GitHub
). If desired, we could set up a simple API key auth in Firecrawl and have Claude use that key (for security).
Resource Considerations: Running headless browser instances can be memory/CPU intensive. Firecrawl’s NUM_WORKERS_PER_QUEUE (in .env) is set to 8 by default
GitHub
. We might reduce that if Jarvis server resources are limited, to avoid running too many parallel browsers. This can be tuned in the env file.
Optional: Tavily Setup: If using Tavily MCP, we would deploy it on a separate port (the open-source tavily-mcp server runs on, say, tcp/5100 by default). The configuration involves an API key from Tavily (for their search index). However, since we assume no cloud, Tavily would be running offline mode – it might still require internet access for search queries. In strictly air-gapped mode, Tavily won’t have an index to query unless we feed it one. Thus, Tavily is optional and useful only if some external search is allowed.
Optional: Brave API: Using Brave’s Search API requires an API key and internet connectivity. If allowed, we could implement a lightweight SearchTool that calls Brave’s endpoint (the Brave API returns web results JSON which we can feed to the agent). If not allowed, skip this.
4. Agent Orchestration with LangGraph
We will implement the agent’s reasoning and tool-usage logic using LangGraph, which is LangChain’s orchestration framework for complex agent workflows. LangGraph enables building agents with structured control flows (graphs) to manage multi-step tasks reliably. We leverage this to integrate memory lookup, RAG retrieval, and tool usage in a controlled manner, rather than leaving all decisions to the LLM.
4.1 Orchestration Workflow Design
The overall flow for each user query will be as follows (each step corresponds to a node in the LangGraph):
Memory Recall Node: On receiving a new query, first retrieve relevant long-term memory:
If using Zep: call Zep’s API (e.g. /api/v1/message_history/search) with keywords from the query or the entire query embedding to fetch related past dialogues or facts. Zep returns any conversation snippets or facts that might be relevant (for example, if the user asks a follow-up, Zep might return the summary of last conversation).
If using LangMem: use the LangMem store (which could be backed by the vector DB) to similarly fetch relevant memory entries by embedding similarity.
Outcome: A short list of memory excerpts (could be none if nothing relevant). These will be appended to the context given to Claude (possibly in a “# Memory” section of the system prompt).
Query Analysis Node (LLM Router): We include an optional LLM-based router that decides which tools or data sources are needed:
For instance, a small prompt to Claude: “Given the user question, decide whether to (a) answer from memory, (b) retrieve docs, (c) search the web. Respond with ‘memory’, ‘docs’, or ‘web’.”
This helps optimize – if the question is purely personal or previously answered, we might not do heavy RAG every time.
However, to ensure comprehensive answers, we might always do docs retrieval unless the router strongly indicates memory suffices.
Graph Retrieval Node: Use GraphRAG capabilities:
Identify any entities in the question (this can be done by simple NER on the query or by prompting Claude in a low-cost way).
Query the Neo4j graph (via Graphiti) for those entities. For example, if the question mentions “API X” and “error Y”, query the graph for node “API X” and find connected nodes like “error Y” or related concepts.
Graphiti’s API can be used here: e.g. GET /entities?name=API X to get an entity ID, then GET /entities/<id>/neighbors to get connected nodes. Alternatively, a Cypher query like MATCH (n)-[r]-(m) WHERE n.name = 'API X' RETURN r,m could be executed.
Outcome: Structured data (or a list of facts) describing relationships. We’ll format it into a text snippet, such as: “KnowledgeGraph: API X is part of Product Z; API X has known issue ‘Error Y’ resolved in v2.3.” This text is added to context. (We may delineate it clearly so Claude knows it’s from the graph.)
Vector Retrieval Node: Perform Vector search in Qdrant:
Generate an embedding for the user’s question (using an embedding model like Llama-2-Embedding or SentenceTransformer locally, or an API like OpenAI if allowed – but aim for local).
Query Qdrant for top N chunks of docs. We will have multiple collections (if we indexed multiple sources) – possibly query each relevant collection (e.g. “product_docs”, “knowledge_base”, etc.).
Retrieve, say, top 3-5 text chunks with their source info.
Outcome: A set of text snippets (e.g. paragraphs from documentation or previous answers) relevant to the query. These are appended to context, prefaced by something like “Reference:” or numbered.
Web Search Node (Optional): If the earlier router or the retrieval steps indicate insufficient info (or if the query is explicitly about something not in our data), use live web search:
Trigger Firecrawl’s search or Brave API to get current information. For example, if user asks about a newly announced feature not yet in docs, or something about “today’s status of …”.
The agent can use an MCP tool called “WebSearch” which we implement to call Firecrawl’s search endpoint or Brave. Claude might say: {"action": "WebSearch", "parameters": {"query": "<user question>"}}. The MCP server will execute that, retrieve say 2-3 result links and feed them back.
Then Claude can invoke another tool to scrape a specific URL (via Firecrawl’s scrape endpoint). For instance, if result 1 looks promising, call Firecrawl /v0/scrape on that URL to get content.
Outcome: Content from the live web. This is potentially slow and expensive, so we do it only if needed. The results are treated as additional context. (They should be marked as such and perhaps limited to e.g. one paragraph to avoid overloading context.)
Answer Generation Node: Finally, with all relevant context gathered (memory, graph facts, doc snippets, and possibly web info), we prompt Claude Code to formulate the answer.
We will construct a final prompt that includes a system message with instructions (and possibly tools available, though at this point tools have already been used via orchestrator – more on MCP integration in next section), the user’s question, and an appended “Context:” section containing the memory and retrieval outputs from steps 1-5.
Claude generates a draft answer.
Post-Answer Processing: After Claude answers, we have a few additional tasks:
Store new information: If the conversation included a new fact (e.g. user provided some info, or the answer itself has important derived knowledge), update memory. For example, send the user’s Q&A pair to Zep for logging, so it can be recalled later.
Rating or Validation: We might include a step where Claude (or another LLM) evaluates its own answer for correctness using retrieved references (this could use a tool like Phoenix’s eval or a simple check for missing citations, etc.). This can be logged for later review but not necessarily blocking response.
Tool feedback loop: If the answer is uncertain or has low confidence, the agent could be designed to attempt a different strategy (this is an advanced option). LangGraph supports adding quality checks and loops. For implementation, we might not do this initially, but keep the architecture open for it (like a node that decides if answer is good enough or if we should try a web search after all).
Finally, return the answer to the user.
All these nodes and transitions form a directed acyclic graph (DAG) in LangGraph. LangGraph’s flexible control flow allows parallel execution too; for instance, steps 3 (graph) and 4 (vector) could run in parallel to save time, then join before generation. We can design the graph accordingly, using LangGraph’s Python API (or YAML workflow if supported). The agent remains controllable – we explicitly moderate its actions (e.g. we only allow certain web endpoints) and can inject human-in-the-loop if needed.
4.2 Claude Code and MCP Integration
Claude Code will act as the reasoning engine, but rather than giving Claude free rein to call tools autonomously, we orchestrate calls via LangGraph. However, to use Model Context Protocol (MCP) effectively, we will integrate the two approaches:
MCP Server: We will run an MCP server locally on Jarvis (e.g. using the official Python SDK or a reference server from the MCP repository). The server will expose a set of tools (data and actions) to the AI and handle JSON-formatted requests from Claude. Think of MCP as a standard interface for tool use.
Registered Tools: We will register connectors for:
Memory – e.g. a MemoryLookup tool that when invoked, calls Zep or LangMem. (We might or might not expose this to Claude directly, since we handle memory in orchestrator. It could be hidden and just used by orchestrator.)
VectorSearch – a tool that takes a query and returns top documents (calls Qdrant).
GraphQuery – a tool that queries the knowledge graph (calls Graphiti or Neo4j).
WebSearch – a tool as described, calling Firecrawl/Brave.
WebScrape – a tool that fetches the content of a URL (via Firecrawl).
Possibly specific utilities, e.g. a Calculator or a Code execution tool, if needed for the agent’s domain (not in scope unless documentation includes code that might be run).
The MCP server runs as a separate process listening on a port (default is 3000 for many examples, but we’ll use maybe 5100 for our custom server to avoid conflict with Firecrawl). Claude (the client) connects to it through its API/SDK.
Claude’s perspective: With MCP, Claude’s prompting would include instructions about available tools and the protocol format. Since in this scenario we largely orchestrate via LangGraph (which itself can call tools directly without going through the language model), we have two possible approaches:
LLM-Driven Agent (MCP-only): We could let Claude handle the decision and calling of tools via MCP messages entirely. Claude would receive a system message enumerating the MCP endpoints (from the server’s self-disclosure), and it would output JSON events to invoke them. Our MCP server would then execute those and provide results back, until Claude decides to respond to the user. This is how an Anthropic tool-enabled agent works by default.
Orchestrator-Driven Agent (LangGraph with MCP hooks): In this approach, Claude is mainly used for reasoning and natural language generation, while LangGraph decides when to call a tool. Essentially, LangGraph nodes make tool calls (API calls) without asking Claude’s permission – they gather data and then feed it to Claude. Claude isn’t aware of the tools per se; it just gets context and produces answers.
Hybrid: We can combine both. For example, LangGraph could handle heavy lifting of retrieval (ensuring certain context is always provided), but also allow Claude to use a limited set of tools if needed for unforeseen queries.
For a reliable implementation, we lean on LangGraph orchestrator to minimize random tool use. We will implement the workflow in code (likely as a Python service or even within the MCP server). The orchestrator can utilize the MCP client internally to call the same connectors. Implementation Plan:
Use the Anthropic Python SDK or API to call Claude (or run a local LLM if needed) within our orchestrator code. We supply the constructed prompt and get the completion.
If we want to keep the door open for Claude to call tools autonomously, we ensure our MCP server is running and consider passing Claude a system message like: “Tools: [SearchDocs, GetGraphInfo, etc.] are available via MCP.” When constructing the final prompt for Claude in step 6 above, we could include a brief list of tools and how to request them (depending on Claude’s formatting requirements for MCP, likely not needed if we directly use the SDK which manages that).
Given that this might complicate prompting, an alternative is to not expose tool use to Claude at all in the prompt, and rely purely on orchestrator to feed it info. This makes Claude’s job simpler (just answer given context).
We will assume Claude has an API or local endpoint we can call. Since everything is local, if we do not have access to Claude API (cloud), we might run a local large model as a substitute (e.g. Llama2 70B via a local server). But since the user specifically said Claude Code, we assume either an API key for Claude or a local instantiation via Claude Desktop that can connect to our MCP. If using Claude Desktop application, the MCP server we run will be connected to Claude through the app (Claude Desktop allows connecting to local MCP servers).
Data Paths & Integration:
The orchestrator service (we can call it Jarvis Orchestrator) will be a container as well, running our Python code. It will mount a volume for any local files (e.g. prompt templates, logs).
This service will connect to:
Zep API (probably http://zep:someport, if Zep exposes an API; in Zep CE, maybe it’s integrated with Graphiti, or it might serve on a separate UI port. We’ll clarify ports in Deployment.)
Qdrant (http://qdrant:6333 or via Qdrant client library).
Neo4j (bolt://neo4j:7687 via Neo4j Python driver).
Firecrawl (http://firecrawl-api:3002).
MCP server (if separate, or orchestrator might itself host MCP endpoints).
We might combine MCP server and orchestrator in one process for simplicity. E.g., using the MCP Python SDK, we can define our tools and directly call our orchestrator’s functions for them. Then launch the MCP server within the orchestrator container. This way, if Claude Desktop connects, it can use the tools as well, and if not, it doesn’t harm the orchestrator’s own operation.
LangGraph Implementation: We will use the LangChain/LangGraph Python library inside the orchestrator:
Define memory using either ZepChatMessageHistory (LangChain has integration for Zep) or a custom long-term memory class.
Define retrievers: a VectorStoreRetriever for Qdrant and perhaps a custom retriever that wraps Neo4j queries (LangChain might have Neo4jGraph support for querying).
Define tools as LangChain tools if letting the agent use them (LangChain has a concept of Tools with name, func, description).
Construct a GraphChain or Workflow that sequences these.
Alternatively, write procedural code using LangGraph primitives: e.g.
with Workflow() as wf:
    memory = FetchMemory(input)
    graph_ctx = QueryGraph(input)
    docs = SearchVectorDB(input)
    context = Combine(memory, graph_ctx, docs)
    answer = LLMAnswer(input + context)
    StoreMemory(input, answer)
    Output(answer)
(Pseudo-code to illustrate.)
LangGraph can manage branches: e.g., if graph_ctx is empty and docs are empty, then branch to WebSearch node, else skip it.
Moderation & Safety: Because the system is self-hosted, we must implement any content moderation if needed. Claude has built-in safety to some extent, but we could also intercept at orchestrator level:
For example, before using a tool, ensure the query is not instructing browsing something disallowed. Firecrawl itself obeys robots.txt and has anti-abuse, but on a closed network likely fine.
We can also log all tool usage for auditing via observability tools (next section).
Data Path Summary:
Memory data: stored in Zep’s Postgres (if it uses one internally) or in Neo4j via Graphiti. Zep container likely includes its own DB or uses files; anyway, persists on disk via volume.
Vector data: persisted in Qdrant volume (we specify one).
Graph data: persisted in Neo4j volume.
Ingestion outputs: ephemeral in pipeline but ultimately land in the above stores.
Inter-service communication: All via Docker network (we will use a network alias so containers can talk by name).
We ensure to define ports for external access where needed:
Claude’s MCP connection: If using Claude Desktop, it connects from host to MCP server on Jarvis. We’ll expose the MCP port (e.g. 5100) to the host.
Firecrawl’s port 3002 could be exposed if we want to manually trigger it from host for debugging (not strictly required for Claude’s usage, but we might expose for convenience).
Neo4j’s 7474 we might expose to use Neo4j Browser interface from a web browser for dev/admin.
Qdrant’s port maybe expose if we want to inspect via REST from host.
Others like Zep’s UI or VNC, see next section.
5. Observability and Evaluation
To monitor the agent’s behavior and continuously improve it, we integrate observability tools:
LangSmith (LangChain Tracking): We can enable LangChain’s CallbackManager with LangSmith to log all prompts, model responses, tool usages, latencies, etc. However, note that self-hosting LangSmith is an enterprise feature (the open-source is not fully available). As an alternative, we might use the LangSmith cloud (not fully local) or skip LangSmith and use open tools as below. Another alternative is LangFuse, an open-source competitor that can self-host similar logging (we mention it as an aside). For now:
We will instrument the agent with OpenTelemetry events for each step. This allows feeding data to Arize Phoenix (which supports OpenTelemetry traces).
Arize Phoenix: Phoenix is an open-source LLM observability and evaluation platform. We include it to monitor and evaluate our agent’s performance:
Phoenix provides tracing, capturing each request/response along with metadata. By using Phoenix’s Python integration (arize-phoenix package), we can log each conversation turn, each tool invocation, and their outcome.
It also supports LLM evaluation, where one model can grade the responses of another for relevance or correctness. We may use this to periodically evaluate answers (though doing it with Claude itself offline might be tricky; Phoenix can use a smaller model or just provide evaluation sets for manual review).
Deployment: We can run Phoenix in a container (arizephoenix/phoenix image). Phoenix’s UI typically runs on a port (say 8080). We will expose it at localhost:8085 (for example) for access via browser. The Phoenix backend can use local files or an in-memory DB to store traces, or connect to their cloud if configured (we’ll keep it local).
The orchestrator will push data to Phoenix either via the Python API (client mode) or via OpenTelemetry traces that Phoenix collects. For simplicity, using the Python client is fine: after each conversation, call phoenix.log_trace(conversation_trace) etc., and we can later open the Phoenix web UI to inspect all interactions, chain of thought, and retrieval relevance.
Phoenix is designed to be framework-agnostic and supports LangChain out-of-the-box. We should leverage that by possibly wrapping our LangGraph calls in LangChain’s trace contexts (which Phoenix can capture).
Weave (W&B Weave): Weave is another toolkit (by Weights & Biases) for LLM app monitoring. It can track inputs/outputs and provide a dashboard. However, W&B Weave typically logs to the W&B cloud by default (though an open-source local deployment might be possible; W&B tends to require login). If strictly no cloud, we might skip Weave or run an on-prem W&B server (which is non-trivial). Instead, we focus on Phoenix, but design the system such that adding Weave later is easy (both are just additional instrumentation).
For completeness: Weave can be integrated by adding a few lines in code (decorators or context managers) to log model calls, similar to LangSmith. If desired, the user could sign in to W&B on Jarvis and have the logs stored in their (cloud or local) W&B instance. We won’t do this by default due to “no cloud” rule.
Logging and Analytics: The Zep UI (if running) provides some analytics about memory usage and knowledge graph state. We will expose it (Zep uses a VNC web interface possibly, see below).
Additionally, we might maintain a simple log file (or database) of all user queries and answers on the filesystem for backup.
We ensure all container logs (stdout/err) are captured (Docker will do so; we might mount a logs/ volume for orchestrator to write custom logs or transcripts).
Evaluation datasets: Over time, one can feed known Q&A pairs into the system and use Phoenix’s evaluation or simple diffing to see if answers improve. The system supports creating datasets of test queries in Phoenix which can be run through the agent periodically to detect regressions.
Observability Deployment: We will include an optional Phoenix service in the compose. Alternatively, since Phoenix can be run as a one-off analysis (like Jupyter mode), we might not keep it always running. But we’ll show it in compose for completeness:
The Phoenix container will likely incorporate a web UI accessible at http://<server>:8080 (we’ll map it to 8085 externally to avoid clash with any other service like perhaps Zep).
Phoenix’s container can store its data in a volume (if it has a database; it might just run in-memory unless configured with persistent storage).
We can configure Phoenix via environment variables if needed (e.g. to point it to traces or to run in offline mode with no cloud). The default pip usage just allows saving and loading trace data from files or not needed if we keep it live.
Summary: With Phoenix (and/or Weave) integrated, we will have:
Real-time view of agent’s internal steps (the chain from memory fetch to final answer) via trace UI.
Ability to evaluate answers against ground truth (if we supply some) or via LLM graders for quality.
System metrics like token usage, latency of each tool, etc., to identify bottlenecks (maybe Firecrawl calls are slow, etc., then we know to adjust).
This closes the loop, enabling continuous improvement of our Jarvis AI system in a self-hosted manner.
6. Deployment Setup and Configuration
We now outline the concrete setup instructions, including Docker Compose definitions, environment configuration, and default data paths/ports for each component. The goal is to provide an implementation-ready guide that Claude Code (or any operator) can follow to launch the system on the Jarvis server.
6.1 System Requirements
Server: The Jarvis server should have sufficient resources (recommend at least 16 GB RAM, 4+ CPU cores, and 100+ GB of disk) especially if running heavy components like a browser or large databases. If running a local LLM model for Claude, ensure a GPU or enough CPU and RAM for inference.
OS: Linux (Docker Engine installed). Docker Compose v2 (or Docker Compose plugin) is required.
Network: No external dependencies required except optional Brave API or Claude API. However, internet access is needed initially to pull Docker images and to crawl documentation (Firecrawl will perform HTTP requests to target sites).
6.2 Directory Layout and Data Paths
We will use a base directory, e.g. /opt/jarvis_ai/ on the server, to organize files:
docker-compose.yml (main orchestration file for all core services).
.env files or config for specific services (if needed).
firecrawl/ directory containing the cloned Firecrawl repo (if building from source).
data/ directory to mount persistent volumes (for Neo4j, Qdrant, etc.):
data/neo4j/ – Neo4j database files.
data/qdrant/ – Qdrant storage.
data/zep/ – (If needed, e.g. if Zep writes any logs or DB, depending on image).
(Redis data we keep ephemeral unless needed; same for Firecrawl ephemeral caches.)
logs/ directory for orchestrator logs or output (optional).
All persistent volumes in Docker Compose will be mapped under ./data for clarity.
6.3 Environment Variables and Secrets
We will define necessary environment variables in a .env file loaded by compose. Key settings:
NEO4J_AUTH=user/password and possibly NEO4J_USER and NEO4J_PASSWORD (for Graphiti).
OPENAI_API_KEY for Graphiti (if using OpenAI for graph operations). If we want to avoid external, we might provide a dummy or, better, configure Graphiti to use a local LLM – Graphiti currently doesn’t support local LLM out-of-box, so we give it an API key to at least a small model (like OpenAI gpt-3.5) for its operations. This is a slight cloud dependency; if unacceptable, we can disable those features of Graphiti (it may still build the graph from provided data without needing to call the LLM).
ZEP_OPENAI_API_KEY (if Zep needs it separately; possibly it uses Graphiti’s key).
CLAUDE_API_KEY if using Claude’s API (if Claude Code is accessible via API and not purely through desktop+MCP). Alternatively, if using Claude locally via desktop MCP, no key needed but we need to ensure Claude Desktop is installed and connected to our MCP.
BRAVE_API_KEY if Brave search is used.
FIRECRAWL_NO_AUTH=true (or similar) to indicate self-host mode with no API key needed (since we disabled DB auth).
Any other optional service keys (ScrapingBee proxy for Firecrawl if needed, etc.) left blank by default.
We store sensitive keys in .env which is not committed anywhere, just on the server.
6.4 Docker Compose Configuration
Below is the Docker Compose configuration for the core system. Each service is defined with necessary settings (image, ports, volumes, environment):
version: '3.8'
services:
  # Vector Database (Qdrant)
  qdrant:
    image: qdrant/qdrant:v1.4.0   # example version
    container_name: jarvis_qdrant
    ports:
      - "6333:6333"    # HTTP API
      - "6334:6334"    # gRPC (if needed)
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped

  # Graph Database (Neo4j)
  neo4j:
    image: neo4j:5.12.0-community
    container_name: jarvis_neo4j
    ports:
      - "7474:7474"    # HTTP browser UI (optional)
      - "7687:7687"    # Bolt protocol for DB
    volumes:
      - ./data/neo4j:/data
    environment:
      - NEO4J_AUTH=${NEO4J_AUTH}   # e.g. "neo4j/secretpassword"
    restart: unless-stopped

  # Graph Memory / GraphRAG Service (Graphiti)
  graphiti:
    image: zepai/graphiti:0.3      # Graphiti server (FastAPI)
    container_name: jarvis_graphiti
    ports:
      - "8000:8000"    # Graphiti HTTP API (internal use, can be closed externally)
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=${NEO4J_USER:-neo4j}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      - neo4j
    restart: unless-stopped

  # Long-term Memory Service (Zep)
  zep:
    image: zepai/zep:1.0
    container_name: jarvis_zep
    # Zep might have a UI via VNC; expose if needed.
    ports:
      - "8080:8080"        # assume Zep UI or API
      - "5901:5901"        # VNC port if using KasmVNC
      - "6901:6901"        # Web VNC port
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ZEP_GRAPHITI_URL=http://graphiti:8000   # (If Zep needs to know Graphiti endpoint)
      # Additional env vars could be set if required by Zep
    depends_on:
      - graphiti
    restart: unless-stopped

  # Redis (for Firecrawl)
  redis:
    image: redis:7-alpine
    container_name: jarvis_redis
    command: ["redis-server", "--save", "", "--appendonly", "no"]  # no persistence needed
    ports:
      - "6379:6379"   # not exposed externally in production, but here for completeness
    restart: unless-stopped

  # Playwright browser service (for Firecrawl, Python version)
  playwright-service:
    build: 
      context: ./firecrawl/apps/playwright-service
      dockerfile: Dockerfile
    container_name: jarvis_playwright
    ports:
      - "3000:3000"   # not exposed externally, internal to Firecrawl
    restart: unless-stopped

  # Firecrawl API service
  firecrawl-api:
    build:
      context: ./firecrawl/apps/api
      dockerfile: Dockerfile
    container_name: jarvis_firecrawl
    env_file: ./firecrawl/.env    # configuration as per Firecrawl's template
    ports:
      - "3002:3002"
    depends_on:
      - redis
      - playwright-service
    restart: unless-stopped

  # Orchestrator & MCP Server (Jarvis Agent Orchestrator)
  orchestrator:
    build: 
      context: ./orchestrator    # assumes we have a Dockerfile to set up LangChain, etc.
      dockerfile: Dockerfile
    container_name: jarvis_orchestrator
    # We assume orchestrator code will start the MCP server internally on 5100
    ports:
      - "5100:5100"   # MCP server port for Claude to connect
    volumes:
      - ./logs:/app/logs        # mount for logs (optional)
      - ./data:/app/data:ro     # read-only access to data if needed (not strictly necessary)
    environment:
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      # (plus any other config orchestrator needs, like endpoints or flags)
    depends_on:
      - qdrant
      - neo4j
      - graphiti
      - zep
      - firecrawl-api
    restart: unless-stopped

  # Observability - Arize Phoenix (optional; can disable if not needed)
  phoenix:
    image: arizephoenix/phoenix:latest
    container_name: jarvis_phoenix
    ports:
      - "8085:8080"   # Phoenix UI (mapped to 8085 on host)
    environment:
      - PHOENIX_API_KEY=   # no key needed for local usage
      # (Phoenix might not need many env vars; it runs an API/UI for logged traces)
    volumes:
      - ./data/phoenix:/phoenix/data   # if Phoenix stores data
    restart: unless-stopped
(Note: The build contexts for Firecrawl require that you have the Firecrawl repository present at ./firecrawl. You should clone the repository and check out a stable release tag. Alternatively, if Mendable provides pre-built images in the future, you can replace build with image: mendableai/firecrawl-api:latest and similarly for playwright. As of writing, self-host requires building from source.) A few points about this configuration:
We used fixed container names (prefixed with jarvis_) for clarity, but it’s optional.
We mapped host ports for services that we may interact with during development:
Qdrant (6333) if we want to manually query it or use its UI.
Neo4j (7474) so we can open Neo4j Browser at http://localhost:7474 (and log in with the credentials we set) to visually inspect the knowledge graph.
Firecrawl API (3002) if we want to trigger crawls manually via curl or Postman from host.
MCP orchestrator (5100) because Claude (if using Claude Desktop) will connect from the host to this port. In production, if Claude runs on same machine or if orchestrator itself calls Claude’s API, this might not be needed.
Phoenix UI (8085 mapped to container 8080).
Zep’s ports: This is speculative; the Zep container might expose a web interface on 8080 (with a KasmVNC environment to show analytics). We exposed 5901/6901 which are typical VNC ports for Kasm images. After starting, one can try connecting to http://localhost:6901 to see if a GUI appears. This interface could show Zep’s analytics dashboard (if implemented). If not needed, these can be removed or changed. Zep might also have a direct REST API on some port, but documentation is sparse – if it uses Graphiti and the Graphiti port, perhaps not.
The orchestrator is built from a local context ./orchestrator. We need to create this:
The Dockerfile would set up a Python environment with needed packages: langchain, langchain-ai (for LangGraph), anthropic (Claude API), neo4j, qdrant-client, arize-phoenix (for logging), etc.
It would copy our orchestrator code (a Python script or package) and run it (maybe via CMD ["python", "-m", "jarvis_orchestrator"]).
The orchestrator on start should initialize the MCP server (listening on 0.0.0.0:5100) and register tools. It can also immediately begin a loop to listen for requests (if using Claude via MCP). If using Claude API, it might instead expose a simple web or CLI interface for user queries (but since user likely uses Claude as interface, we focus on MCP route).
In our design, we assume the user will chat with Claude through the Claude app or interface, and the Claude -> MCP -> orchestrator pathway handles queries. We could also have orchestrator open a local chat UI or CLI as an alternative interface.
After creating the above file, run the deployment with:
# Navigate to the base directory
cd /opt/jarvis_ai

# Ensure the Firecrawl repository is present for building images
git clone https://github.com/mendableai/firecrawl.git firecrawl
# (Optional) checkout a specific release known to be stable
# git checkout tags/v1.15.0  (if such tag exists, for example)

# Copy environment template for Firecrawl
cp firecrawl/apps/api/.env.example firecrawl/.env
# Edit firecrawl/.env to set at least:
#   NUM_WORKERS_PER_QUEUE=4 (for example, to reduce load)
#   HOST=0.0.0.0, PORT=3002 (already default)
#   REDIS_URL=redis://redis:6379 (for docker setup)
#   PLAYWRIGHT_MICROSERVICE_URL=http://playwright-service:3000/html
#   USE_DB_AUTHENTICATION=false, and all API keys blank for now.

# Create orchestrator code and Dockerfile (as per Section 6.5 below).

# Launch all services
docker-compose up -d
This will pull/build images and start containers in the background. Use docker-compose logs -f orchestrator to monitor the orchestrator startup, and ensure each service comes up without errors:
Neo4j will print logs indicating it’s ready and Bolt connector is listening.
Graphiti will log that it’s connected to Neo4j (and maybe waiting for queries).
Zep might take longer to start; look for any log that it’s ready or listening.
Firecrawl will log the API endpoint (you should see something like “Server listening on port 3002”).
Orchestrator will likely log the MCP server startup (e.g. “MCP server listening at 0.0.0.0:5100”).
Phoenix will log a message about its web interface being available.
Check that:
Qdrant’s storage directory ./data/qdrant is created.
Neo4j’s data directory has files (and you can open http://<server>:7474 in a browser, it should ask for user/pass which you set as env).
Firecrawl is accessible: try curl http://localhost:3002/v0/health (if such exists) or check docker-compose logs firecrawl-api for something like “Listening on 3002”.
The orchestrator doesn’t crash. If it’s waiting for connections (Claude via MCP), it might just sit quietly after initialization.
6.5 Orchestrator Implementation Notes
In the orchestrator code (which is beyond the scope of this text to fully write out), we will do roughly:
Initialize LangChain/LangGraph components:
Load our API keys from env (for Claude, etc.).
Create a ConversationBufferMemory if short-term needed (though we lean on long-term memory).
Create a ZepMemory object (LangChain provides one) if using Zep, configured with Zep’s base URL (likely http://zep:8000 or another port if Zep has an API).
Initialize Qdrant client: e.g. qdrant_client = QdrantClient(url="http://qdrant:6333").
Initialize Neo4j driver: e.g. using NEO4J_URI, NEO4J_USER/PASS from env.
Define a retrieval function for docs: e.g. vector_retriever = QdrantRetriever(qdrant_client, collection_name="docs", embedding_model=embedding_fn).
Define graph query function: can be direct Cypher or using Graphiti’s REST. For simplicity, maybe use Graphiti REST by calling requests.get("http://graphiti:8000/query", params={...}) if such exists. If not, directly use neo4j_driver.session().run(<cypher>).
Set up MCP server via modelcontext library:
e.g. server = MCPServer(host="0.0.0.0", port=5100)
Register tools:
@server.tool(name="SearchDocs", description="...") def search_docs(query: str): ... that uses vector_retriever and returns text.
Similarly @server.tool(name="GetGraphInfo", ...) def get_graph(entity: str): ... that queries Neo4j for that entity’s neighbors and returns a summary.
@server.tool(name="WebSearch", ...) def web_search(query: str): ... that calls Firecrawl’s search and returns result titles.
@server.tool(name="WebScrape", ...) def web_scrape(url: str): ... that calls Firecrawl’s scrape and returns text.
Start the server: server.run() or similar.
MCP Interaction: If using Claude via the desktop app, the app will handle sending the conversation to MCP. We ensure our tools produce concise JSON outputs or text that Claude can parse.
If using Claude API programmatically (instead of the desktop UI), we would have to implement the loop of sending Claude a prompt, getting output, and checking for tool invocation format (MCP spec uses a specific message structure). This is more complex and likely not needed if we use Claude’s built-in support via the desktop. So we assume the primary interface is Claude with MCP.
Usage Note: To use the system, one would:
Open Claude (Desktop) on their machine, connect it to the MCP server (http://<jarvis_server>:5100 as per Anthropic’s guide).
Then in Claude’s chat, ask questions. Claude will then behind the scenes call MCP tools. For example:
User asks: “What does the API endpoint /v1/orders do in Acme’s service? And was there any change in 2023?”
Claude will perhaps call SearchDocs with query "API /v1/orders Acme 2023 change" via MCP.
Our orchestrator executes that: Qdrant returns a doc snippet saying “/v1/orders: create new order. Changed in v2.0 (2023) – requires an API key.”
Claude gets that result, then maybe it calls GetGraphInfo with entity "Acme API v1/orders". The graph might have a node for that endpoint linked to a "version 2.0 (2023)" node.
Orchestrator returns: “/v1/orders --updated-> 2023-05-01 (version 2.0 release)” for example.
Claude now has enough and formulates an answer about the API and its change in 2023.
Finally, orchestrator logs all this to Phoenix; we can later see the tool calls timeline and the answer.
6.6 Persistent Storage and Backup
The system’s knowledge and state reside in:
Neo4j graph (persisted under ./data/neo4j).
Qdrant vector store (./data/qdrant).
Zep memory (possibly in an internal SQLite or files in its container – if so, mount ./data/zep if needed; we might adjust the compose to volume-mount a data dir if we find where Zep stores community edition data).
Logs in ./logs and Phoenix data in ./data/phoenix (if used).
Regular backups of these volumes are recommended. Neo4j can be backed up via its dump utilities or by copying the database files when the container is stopped. Qdrant data can be copied as well (it’s usually just a directory of segment files). Since everything is local, one could also snapshot the entire data/ folder.
6.7 Security and Access
Because everything is on a single server, firewall the ports as needed:
Expose MCP (5100) only to the users who need to connect (if server is within a LAN, ensure it’s not open to internet).
Similarly, Firecrawl (3002) and Neo4j UI (7474) could be sensitive; limit access via firewall or Docker’s network settings (we could remove the ports: mapping for 3002 and 7474 if not needed externally – the orchestrator and Claude (via MCP server) can still reach them on the internal network).
If multiple people use the system, consider enabling authentication:
Firecrawl supports API keys (via Supabase integration, though not trivial offline).
MCP server could be behind an SSH tunnel or VPN if needed.
Neo4j has its user/pass; use a strong password there.
Claude’s outputs should be monitored if connected to the internet (in our design, minimal external calls are made).
The server should be secured (since we run a browser via Firecrawl that might fetch untrusted content, ensure OS packages (like Chrome in Playwright) are up-to-date to avoid exploits. Firecrawl’s container isolates that, but still good practice).
6.8 Testing and Usage Notes
Once setup:
Test memory: Ask something that was mentioned in a previous session to see if Zep recalls it.
Test doc retrieval: Ask a question from ingested docs and ensure the answer cites the correct info (if we program Claude to cite sources, which we can – we might instruct Claude to include source names if desired).
Test graph: Ask a question requiring a relationship, e.g. “Which company makes Product X and what other products do they have?” If our graph has that, see if answer comes from graph data.
Check Phoenix: Run a few queries, then go to http://localhost:8085 and explore the traces. You should see events for each tool call and the prompt/response. Phoenix can also highlight any hallucinations if we feed it expected answers for some queries (advanced usage).
The design above aims for modularity: each component (memory, vector DB, graph, etc.) can be replaced or upgraded independently. For instance, if you prefer Weaviate over Qdrant, you can swap the service and adjust the orchestrator’s vector retrieval code. If you want to try a different LLM, you could connect the MCP to an open-source model (e.g. through OpenAI’s function calling interface or LangChain Agents). Finally, ensure to update documentation as needed and monitor resource usage (the Playwright browser can be heavy; in low-memory environment consider reducing NUM_WORKERS_PER_QUEUE to 1 or 2 so only one headless browser instance runs at a time).
Sources:
LangMem introduction (long-term memory for LLMs)\
Zep memory platform description\
Zep Community Edition (graph-based memory) announcement\
GraphRAG concept and benefits of hybrid approach\
Apify blog on Firecrawl vs Tavily (web ingestion approaches)\
Firecrawl architecture (crawling, headless browser)\
LangGraph orchestration benefits (control flows, memory)\
Model Context Protocol intro (standardizing AI-tool integration)\
Arize Phoenix features (tracing and evaluation for LLM apps)\
W&B Weave purpose (LLM app monitoring toolkit)


Bronnen



Geen bestand gekozenGeen bestand gekozen
ChatGPT kan fouten maken. OpenAI gebruikt geen werkruimtegegevens van Threatintel om modellen te trainen.

Design Document: Self-Hosted Agentic AI System on Jarvis Server
Overview and Architecture
This document describes a fully self-hosted agentic AI system for the Jarvis server. The system integrates long-term memory, a hybrid Retrieval-Augmented Generation (RAG) knowledge stack (combining vector search and knowledge graphs), live web data ingestion, orchestrated tool use, and observability – all running locally (no cloud dependencies). The AI interface is Claude Code, connected via the Model Context Protocol (MCP) to access these local tools and data. Key Components:
Persistent Memory Layer: Long-term episodic and semantic memory using either LangMem SDK or a self-hosted Zep memory service.
Hybrid RAG Stack: A vector database (e.g. Qdrant or Weaviate) for semantic search, combined with a Neo4j graph database for GraphRAG (knowledge graph retrieval).
Live Web Ingestion: Automated crawling of documentation and web content using Firecrawl (self-hosted) with optional integration of Tavily search (MCP server) or the Brave Search API for real-time grounding.
Agent Orchestration: A workflow defined with LangGraph, enabling controlled multi-step tool use and integration of memory/retrieval in the agent’s reasoning.
AI Interface (LLM): Claude Code acting as the language model, connected to local tools via an MCP server. Claude’s prompts and outputs are enriched with relevant context fetched through MCP tool calls.
Observability & Evaluation: Instrumentation via LangSmith (LangChain’s tracing) and/or open-source observability tools like Arize Phoenix and W&B Weave for monitoring, debugging, and evaluating the AI agent’s performance.
Below we detail each component’s design and implementation. All services will be containerized (Docker) for ease of deployment, with persistent storage for data. We also provide setup instructions, configuration specifics, and notes on how Claude Code can use this setup.
1. Persistent Memory Layer (Episodic & Semantic Memory)
Objective: Enable the AI agent to remember past interactions and learned facts over long-term sessions. We implement persistent memory via either LangMem (in-code SDK) or Zep (self-hosted memory server).
LangMem (SDK Option): LangMem is an open-source long-term memory SDK for LLM agents. It extracts and stores important information from conversations, complementing short-term context with a growing knowledge base. If using LangMem, the agent’s code (LangGraph workflow) will incorporate:
Episodic memory: storing raw conversation transcripts or summaries after each session (e.g. in a local database or file).
Semantic memory: extracting key facts or embeddings from conversations and storing them in the vector store (Qdrant/Weaviate) for future retrieval.
LangMem provides patterns for semantic memory extraction (e.g. summarizing each interaction and indexing it). These functions would be called in the agent pipeline after each user turn.
Integration: LangMem can interface with the vector DB (to store embeddings of memories) and with the graph DB if we choose to log certain facts as nodes/edges. This requires writing Python code in the LangGraph workflow to call LangMem’s APIs.
Zep (Server Option): Zep is a dedicated long-term memory service that learns from interactions and builds a temporal knowledge graph of facts. We will use Zep Community Edition (open-source) self-hosted:
Zep stores conversation history, auto-summarizes it, performs vector indexing of memory, and even tracks factual changes over time via an internal knowledge graph.
We deploy Zep as a Docker service (see Deployment section for zep container). Zep connects to a Neo4j database and a graph service (Graphiti) under the hood to maintain the knowledge graph of memories.
Episodic memory: The agent (Claude) will send conversation transcripts to Zep via its API after each interaction. Zep will persist the raw messages and generate summaries.
Semantic memory: Zep enriches stored chats with extracted facts, embeddings, and a knowledge graph of entities (using Graphiti). This allows later retrieval of relevant facts by context or by structured queries.
Integration: The LangGraph agent can query Zep’s API for relevant memory when a new user query arrives (e.g. “retrieve last N interactions and any facts about X from memory”). Zep’s temporal knowledge graph can answer queries like “what product did the user mention previously?” with low latency.
Zep requires an LLM for its summarization/extraction tasks; by default it expects an OpenAI API (we will configure an open-source model or ensure this call is local to avoid cloud – see Configuration below).
Choice: Either approach yields persistent memory. For a fully self-hosted stack, we will proceed with Zep as it provides a ready-made service including memory search and graph capabilities. (If Zep is not desired, one can use LangMem with a custom implementation – in that case skip the Zep container and include LangMem calls in the orchestrator code.)
2. Hybrid Retrieval-Augmented Generation (RAG) Stack
Objective: Enhance the agent’s knowledge access by combining vector-based retrieval of unstructured data and graph-based retrieval of structured relationships (GraphRAG). This hybrid approach improves both recall and precision of context provided to the LLM.
2.1 Vector Database for Semantic Search
We deploy a vector database to store embeddings of documents, knowledge snippets, and semantic memory entries for similarity search. Options are Qdrant or Weaviate, both of which are open-source and self-hostable:
Qdrant: A high-performance vector store, ideal for semantic search at scale. We’ll use Qdrant in this design (containerized via official image). Qdrant listens on port 6333 for API (gRPC/REST). It will store embeddings (e.g. OpenAI Ada or local model) of our ingested vendor docs and any additional data.
Weaviate: Alternatively, Weaviate could be used similarly (with a container on port 8080). Weaviate has modular components for text, but running it offline may require disabling its cloud modules. For simplicity we proceed with Qdrant.
Usage: The agent will use vector search to retrieve relevant text passages given a user query:
The LangGraph orchestrator (or Claude via a tool call) will take the user’s question, generate an embedding (using a local embedding model or an API), and query Qdrant for nearest vectors.
Qdrant returns the most similar documents or snippets. These will be fed into Claude’s context as part of the prompt (often referred to as retrieved context).
This provides the LLM with detailed reference material to ground its answer.
We will configure a persistent volume for Qdrant to store vectors on disk (so data persists across restarts). The collection name and vector dimensionality must be set (e.g. 1536 if using OpenAI embeddings). The orchestration code will handle inserting new embeddings when new docs are ingested or new memory items are created.
2.2 Graph Database for Knowledge Graph (GraphRAG)
We deploy Neo4j (v5.x Community Edition) as a local graph database to store a knowledge graph of facts and their relationships, supporting the GraphRAG approach
qdrant.tech
. Neo4j is well-suited for representing entities (nodes) and relationships (edges) that the agent can query to answer complex questions. Use Cases:
Store structured information extracted from documents (e.g. product A <–belongs_to– Company B, API X <–requires– Permission Y, etc.).
Store key facts from conversations or user profile in a way that relationships can be traversed (e.g. User <–prefers– Product Z).
The agent can perform graph queries (via a Neo4j query tool or via an MCP integration) to find connected information relevant to a query. For example: “What is the latest version of Library L and when was it released?” could be answered by traversing Library L node to a Version node, rather than relying solely on fuzzy text search.
GraphRAG Integration: GraphRAG enhances RAG by using the knowledge graph context. A typical pattern:
When ingesting documents, we not only embed the text to vectors, but also parse and identify entities and relations, inserting them into Neo4j.
At query time, the agent or orchestrator first uses an entity linker (or prompt) to identify key entities in the question, and queries Neo4j for their relationships or properties.
The result is a subgraph of relevant facts. This graph context can be converted into text or a structured format and provided to the LLM.
Simultaneously, a traditional vector search is done for additional context. The combination of graph-derived facts and semantically similar text gives a comprehensive context.
For example, Microsoft’s GraphRAG approach uses a knowledge graph to provide deterministic context for queries, following chains of relationships for complex reasoning
qdrant.tech
. In our system, Neo4j will serve this role:
We will run Neo4j in a Docker container (with Bolt protocol on port 7687 and HTTP interface on 7474 for browser access if needed). A volume will persist its data/ directory.
We set an admin user and password (in the compose file or via env NEO4J_AUTH) to secure the DB.
The Graphiti service (described below) and/or our orchestrator will connect to Neo4j to read/write graph data.
Graph Construction: During ingestion (Section 3), after crawling documents, we can use simple parsing or LLM-based extraction to create triples for the graph:
We might define a schema of node labels and relationship types (for example: Product, Company, Feature, etc., with relationships like BELONGS_TO, DEPENDS_ON, VERSION_OF, etc.).
Using an LLM in a local capacity (or rules based on doc structure), we extract these relations and push to Neo4j via Cypher queries.
This can be automated in code or even using Graphiti, which as part of Zep can build a temporal knowledge graph from JSON or text data.
Graphiti (Knowledge Graph MCP Server): Zep’s open-source ecosystem includes Graphiti, a service that maintains and queries the temporal knowledge graph. We will deploy Graphiti as a separate container that works with Neo4j:
Graphiti (container graphiti) runs a FastAPI server (on port 8000) that exposes endpoints for upserting facts into the graph and retrieving subgraphs relevant to queries.
Graphiti requires connection info for Neo4j (provided via environment variables). It also requires an OpenAI API key to perform some LLM-driven tasks (like generating graph nodes or answering in natural language). We will configure it to use our local Claude or another model via MCP or skip any cloud calls (see Config).
In our architecture, Graphiti serves two purposes:
Internal Memory Graph: If using Zep, Graphiti is used by Zep to store conversation facts.
General Knowledge Graph: We can also use Graphiti’s API ourselves to store and query data from ingested docs (not just conversation). For example, we can call Graphiti to insert an entity and relationships parsed from documentation.
Graphiti is essentially an MCP server for knowledge graphs – it provides a standardized API that an AI agent can call to query graph data. We can register Graphiti as a tool in the MCP so that Claude can directly ask for graph info (e.g. “find relations of X”).
Graph Querying: The agent can either:
Use Cypher queries via a tool: We could expose a custom tool that executes a Cypher query on Neo4j and returns results (this might be risky to let the LLM form Cypher; safer to have preset query patterns).
Use Graphiti’s higher-level endpoints: e.g. “get neighbors of node matching X” or “find shortest path between A and B” etc. This is more structured and easier to use via MCP tools with fixed parameters.
In summary, the RAG stack comprises Qdrant for semantic similarity and Neo4j/Graphiti for knowledge graph retrieval. This hybrid approach (GraphRAG) “combines a Graph Database and a Vector Database to improve RAG workflows”, providing both deep structured knowledge and broad unstructured context to Claude.
3. Live Web Ingestion Pipeline
Objective: Continuously or on-demand ingest fresh content (especially vendor documentation, knowledge base articles, etc.) into the system’s knowledge stores. This ensures Claude always has up-to-date data from external sources, despite being self-contained. We implement a web ingestion pipeline with the following components and steps:
3.1 Firecrawl – Self-Hosted Web Crawler & Scraper
Firecrawl is a tool that, given a starting URL, can crawl all accessible pages and extract their content in an LLM-friendly format (Markdown or JSON). We will self-host Firecrawl to ingest documentation websites. Firecrawl provides a unified API to handle scraping (basic fetch) or headless browsing when needed, and it automatically cleans the content (removing boilerplate, etc.). Some Firecrawl highlights:
It navigates internal links of a site (respecting scope) without needing a sitemap, skipping duplicates and focusing on relevant pages.
A decision layer chooses between a simple fetch vs. a headless browser (Chromium via Playwright) depending on the page’s needs (static vs dynamic).
It can solve simple interactions (through a built-in agent called FIRE-1) for things like “Load More” buttons or basic captchas.
It outputs content in Markdown (reducing token count significantly by stripping HTML), or structured JSON if using the extract mode with a schema.
Deployment: Firecrawl’s self-hosted setup consists of multiple services:
The API service (firecrawl-api) – the main REST interface (running on port 3002) that accepts crawl/scrape/search requests.
A Redis instance – used by Firecrawl for task queueing and rate limiting.
A Playwright browser service – a microservice that Firecrawl API calls for rendering JavaScript-heavy pages. Firecrawl provides a default Playwright service (we will run it).
(Optional) Additional microservices for special features (not required for basic crawling).
We will deploy Firecrawl via Docker Compose using the open-source repository:
Clone the Firecrawl repository (mendableai/firecrawl) to Jarvis. Copy the example .env from apps/api/.env.example to configure the environment. Key settings:
PORT=3002 (API service port), HOST=0.0.0.0.
REDIS_URL=redis://redis:6379 (pointing to our Redis container).
PLAYWRIGHT_MICROSERVICE_URL=http://playwright-service:3000/html (URL for the browser renderer).
Disable any cloud-specific features: USE_DB_AUTHENTICATION=false (we won’t use Supabase auth for self-hosted), leave optional API keys blank (we won’t use external OCR or Slack, etc., for now).
Use the provided docker-compose.yml (or integrate into our main compose) to bring up:
redis (Redis 7 alpine),
playwright-service (Node or Python service for rendering – default to Python service),
firecrawl-api service (which depends on the above).
Networking: We will keep these services on the same Docker network as the rest of the system so our orchestrator or MCP server can call Firecrawl’s API at http://firecrawl-api:3002.
Usage Workflow:
Scheduling: We set up a schedule (e.g. a cron job or a recurring container task) to periodically crawl target docs. For instance, every night at 2 AM, call Firecrawl’s /v0/crawl endpoint with the base URL of a vendor’s docs site.
Crawling: Firecrawl will fetch all pages (respecting depth limits we can configure) and return the content. For large sites, the crawl endpoint can optionally deliver results via webhook or we page through results. In our setup, since we have direct access, the orchestrator may call Firecrawl and process results in streaming fashion.
Storage: The content returned (in Markdown or text) is then processed:
For each page, generate an embedding and upsert into Qdrant (vector DB) under a collection for that documentation.
Extract structured facts (if possible) using simple rules or an LLM. For example, identify headings, API method names, version numbers, etc., and create graph nodes/edges in Neo4j. (This could be done by a small parsing script or by prompting Claude to output triples from the text, then inserting them.)
Optionally, store the raw text or markdown in a local document store (even just files on disk or a simple SQLite full-text index) for backup or direct retrieval.
Verification: Log or output the status of ingestion (e.g. number of pages, any errors). Firecrawl provides an admin UI at http://localhost:3002/admin/@/queues to monitor jobs
GitHub
.
On-Demand Updates: In addition to scheduled full crawls, we can use Firecrawl’s search capability for real-time needs. Firecrawl’s /v0/search can perform a live web search (it uses Bing or other sources under the hood) and then scrape each result. However, since we avoid cloud, this might rely on public search endpoints. Alternatively:
Brave Search API: Brave offers an API for web search results suitable for grounding LLMs. If the user’s query is outside the ingested knowledge, Claude could call a “WebSearch” tool (via MCP) that hits the Brave API and returns top results, which Firecrawl can then scrape for content. This keeps the agent up-to-date with latest info.
Tavily MCP Server: Tavily is another tool that focuses on real-time search-first web access. Tavily has an open-source MCP server (MIT licensed) that mirrors its hosted API. We can optionally deploy the Tavily MCP server if we want a ready-made solution for combined web search + extraction:
Tavily’s approach: first perform a multi-vertical search (with relevance scoring) then use an extract call to get the content of chosen results.
It can complement Firecrawl by quickly finding relevant pages across the web, whereas Firecrawl is excellent for crawling a known domain in-depth.
If deployed, Claude (via MCP) can use Tavily’s server as a SearchTool (like a self-hosted alternative to Googling).
Note: Tavily MCP server and Firecrawl overlap in functionality. We likely choose one primary (Firecrawl) and perhaps include Tavily as an alternative if needed. The system is designed to support both, not necessarily run both simultaneously unless needed.
In summary, Firecrawl will ensure our local vector and graph databases are populated with the latest documentation content. We will implement a daily ingestion loop (using cron or a simple Python scheduler service) that calls Firecrawl and updates the stores. This keeps knowledge fresh without relying on external APIs at query time.
3.2 Configuration for Firecrawl & Alternatives
Firecrawl Access: After deployment, the Firecrawl API will be accessible at http://<server>:3002. We will not expose this publicly; it’s only for local use by the AI. The .env for Firecrawl is configured with no auth (USE_DB_AUTHENTICATION=false means API keys are not enforced
GitHub
GitHub
). If desired, we could set up a simple API key auth in Firecrawl and have Claude use that key (for security).
Resource Considerations: Running headless browser instances can be memory/CPU intensive. Firecrawl’s NUM_WORKERS_PER_QUEUE (in .env) is set to 8 by default
GitHub
. We might reduce that if Jarvis server resources are limited, to avoid running too many parallel browsers. This can be tuned in the env file.
Optional: Tavily Setup: If using Tavily MCP, we would deploy it on a separate port (the open-source tavily-mcp server runs on, say, tcp/5100 by default). The configuration involves an API key from Tavily (for their search index). However, since we assume no cloud, Tavily would be running offline mode – it might still require internet access for search queries. In strictly air-gapped mode, Tavily won’t have an index to query unless we feed it one. Thus, Tavily is optional and useful only if some external search is allowed.
Optional: Brave API: Using Brave’s Search API requires an API key and internet connectivity. If allowed, we could implement a lightweight SearchTool that calls Brave’s endpoint (the Brave API returns web results JSON which we can feed to the agent). If not allowed, skip this.
4. Agent Orchestration with LangGraph
We will implement the agent’s reasoning and tool-usage logic using LangGraph, which is LangChain’s orchestration framework for complex agent workflows. LangGraph enables building agents with structured control flows (graphs) to manage multi-step tasks reliably. We leverage this to integrate memory lookup, RAG retrieval, and tool usage in a controlled manner, rather than leaving all decisions to the LLM.
4.1 Orchestration Workflow Design
The overall flow for each user query will be as follows (each step corresponds to a node in the LangGraph):
Memory Recall Node: On receiving a new query, first retrieve relevant long-term memory:
If using Zep: call Zep’s API (e.g. /api/v1/message_history/search) with keywords from the query or the entire query embedding to fetch related past dialogues or facts. Zep returns any conversation snippets or facts that might be relevant (for example, if the user asks a follow-up, Zep might return the summary of last conversation).
If using LangMem: use the LangMem store (which could be backed by the vector DB) to similarly fetch relevant memory entries by embedding similarity.
Outcome: A short list of memory excerpts (could be none if nothing relevant). These will be appended to the context given to Claude (possibly in a “# Memory” section of the system prompt).
Query Analysis Node (LLM Router): We include an optional LLM-based router that decides which tools or data sources are needed:
For instance, a small prompt to Claude: “Given the user question, decide whether to (a) answer from memory, (b) retrieve docs, (c) search the web. Respond with ‘memory’, ‘docs’, or ‘web’.”
This helps optimize – if the question is purely personal or previously answered, we might not do heavy RAG every time.
However, to ensure comprehensive answers, we might always do docs retrieval unless the router strongly indicates memory suffices.
Graph Retrieval Node: Use GraphRAG capabilities:
Identify any entities in the question (this can be done by simple NER on the query or by prompting Claude in a low-cost way).
Query the Neo4j graph (via Graphiti) for those entities. For example, if the question mentions “API X” and “error Y”, query the graph for node “API X” and find connected nodes like “error Y” or related concepts.
Graphiti’s API can be used here: e.g. GET /entities?name=API X to get an entity ID, then GET /entities/<id>/neighbors to get connected nodes. Alternatively, a Cypher query like MATCH (n)-[r]-(m) WHERE n.name = 'API X' RETURN r,m could be executed.
Outcome: Structured data (or a list of facts) describing relationships. We’ll format it into a text snippet, such as: “KnowledgeGraph: API X is part of Product Z; API X has known issue ‘Error Y’ resolved in v2.3.” This text is added to context. (We may delineate it clearly so Claude knows it’s from the graph.)
Vector Retrieval Node: Perform Vector search in Qdrant:
Generate an embedding for the user’s question (using an embedding model like Llama-2-Embedding or SentenceTransformer locally, or an API like OpenAI if allowed – but aim for local).
Query Qdrant for top N chunks of docs. We will have multiple collections (if we indexed multiple sources) – possibly query each relevant collection (e.g. “product_docs”, “knowledge_base”, etc.).
Retrieve, say, top 3-5 text chunks with their source info.
Outcome: A set of text snippets (e.g. paragraphs from documentation or previous answers) relevant to the query. These are appended to context, prefaced by something like “Reference:” or numbered.
Web Search Node (Optional): If the earlier router or the retrieval steps indicate insufficient info (or if the query is explicitly about something not in our data), use live web search:
Trigger Firecrawl’s search or Brave API to get current information. For example, if user asks about a newly announced feature not yet in docs, or something about “today’s status of …”.
The agent can use an MCP tool called “WebSearch” which we implement to call Firecrawl’s search endpoint or Brave. Claude might say: {"action": "WebSearch", "parameters": {"query": "<user question>"}}. The MCP server will execute that, retrieve say 2-3 result links and feed them back.
Then Claude can invoke another tool to scrape a specific URL (via Firecrawl’s scrape endpoint). For instance, if result 1 looks promising, call Firecrawl /v0/scrape on that URL to get content.
Outcome: Content from the live web. This is potentially slow and expensive, so we do it only if needed. The results are treated as additional context. (They should be marked as such and perhaps limited to e.g. one paragraph to avoid overloading context.)
Answer Generation Node: Finally, with all relevant context gathered (memory, graph facts, doc snippets, and possibly web info), we prompt Claude Code to formulate the answer.
We will construct a final prompt that includes a system message with instructions (and possibly tools available, though at this point tools have already been used via orchestrator – more on MCP integration in next section), the user’s question, and an appended “Context:” section containing the memory and retrieval outputs from steps 1-5.
Claude generates a draft answer.
Post-Answer Processing: After Claude answers, we have a few additional tasks:
Store new information: If the conversation included a new fact (e.g. user provided some info, or the answer itself has important derived knowledge), update memory. For example, send the user’s Q&A pair to Zep for logging, so it can be recalled later.
Rating or Validation: We might include a step where Claude (or another LLM) evaluates its own answer for correctness using retrieved references (this could use a tool like Phoenix’s eval or a simple check for missing citations, etc.). This can be logged for later review but not necessarily blocking response.
Tool feedback loop: If the answer is uncertain or has low confidence, the agent could be designed to attempt a different strategy (this is an advanced option). LangGraph supports adding quality checks and loops. For implementation, we might not do this initially, but keep the architecture open for it (like a node that decides if answer is good enough or if we should try a web search after all).
Finally, return the answer to the user.
All these nodes and transitions form a directed acyclic graph (DAG) in LangGraph. LangGraph’s flexible control flow allows parallel execution too; for instance, steps 3 (graph) and 4 (vector) could run in parallel to save time, then join before generation. We can design the graph accordingly, using LangGraph’s Python API (or YAML workflow if supported). The agent remains controllable – we explicitly moderate its actions (e.g. we only allow certain web endpoints) and can inject human-in-the-loop if needed.
4.2 Claude Code and MCP Integration
Claude Code will act as the reasoning engine, but rather than giving Claude free rein to call tools autonomously, we orchestrate calls via LangGraph. However, to use Model Context Protocol (MCP) effectively, we will integrate the two approaches:
MCP Server: We will run an MCP server locally on Jarvis (e.g. using the official Python SDK or a reference server from the MCP repository). The server will expose a set of tools (data and actions) to the AI and handle JSON-formatted requests from Claude. Think of MCP as a standard interface for tool use.
Registered Tools: We will register connectors for:
Memory – e.g. a MemoryLookup tool that when invoked, calls Zep or LangMem. (We might or might not expose this to Claude directly, since we handle memory in orchestrator. It could be hidden and just used by orchestrator.)
VectorSearch – a tool that takes a query and returns top documents (calls Qdrant).
GraphQuery – a tool that queries the knowledge graph (calls Graphiti or Neo4j).
WebSearch – a tool as described, calling Firecrawl/Brave.
WebScrape – a tool that fetches the content of a URL (via Firecrawl).
Possibly specific utilities, e.g. a Calculator or a Code execution tool, if needed for the agent’s domain (not in scope unless documentation includes code that might be run).
The MCP server runs as a separate process listening on a port (default is 3000 for many examples, but we’ll use maybe 5100 for our custom server to avoid conflict with Firecrawl). Claude (the client) connects to it through its API/SDK.
Claude’s perspective: With MCP, Claude’s prompting would include instructions about available tools and the protocol format. Since in this scenario we largely orchestrate via LangGraph (which itself can call tools directly without going through the language model), we have two possible approaches:
LLM-Driven Agent (MCP-only): We could let Claude handle the decision and calling of tools via MCP messages entirely. Claude would receive a system message enumerating the MCP endpoints (from the server’s self-disclosure), and it would output JSON events to invoke them. Our MCP server would then execute those and provide results back, until Claude decides to respond to the user. This is how an Anthropic tool-enabled agent works by default.
Orchestrator-Driven Agent (LangGraph with MCP hooks): In this approach, Claude is mainly used for reasoning and natural language generation, while LangGraph decides when to call a tool. Essentially, LangGraph nodes make tool calls (API calls) without asking Claude’s permission – they gather data and then feed it to Claude. Claude isn’t aware of the tools per se; it just gets context and produces answers.
Hybrid: We can combine both. For example, LangGraph could handle heavy lifting of retrieval (ensuring certain context is always provided), but also allow Claude to use a limited set of tools if needed for unforeseen queries.
For a reliable implementation, we lean on LangGraph orchestrator to minimize random tool use. We will implement the workflow in code (likely as a Python service or even within the MCP server). The orchestrator can utilize the MCP client internally to call the same connectors. Implementation Plan:
Use the Anthropic Python SDK or API to call Claude (or run a local LLM if needed) within our orchestrator code. We supply the constructed prompt and get the completion.
If we want to keep the door open for Claude to call tools autonomously, we ensure our MCP server is running and consider passing Claude a system message like: “Tools: [SearchDocs, GetGraphInfo, etc.] are available via MCP.” When constructing the final prompt for Claude in step 6 above, we could include a brief list of tools and how to request them (depending on Claude’s formatting requirements for MCP, likely not needed if we directly use the SDK which manages that).
Given that this might complicate prompting, an alternative is to not expose tool use to Claude at all in the prompt, and rely purely on orchestrator to feed it info. This makes Claude’s job simpler (just answer given context).
We will assume Claude has an API or local endpoint we can call. Since everything is local, if we do not have access to Claude API (cloud), we might run a local large model as a substitute (e.g. Llama2 70B via a local server). But since the user specifically said Claude Code, we assume either an API key for Claude or a local instantiation via Claude Desktop that can connect to our MCP. If using Claude Desktop application, the MCP server we run will be connected to Claude through the app (Claude Desktop allows connecting to local MCP servers).
Data Paths & Integration:
The orchestrator service (we can call it Jarvis Orchestrator) will be a container as well, running our Python code. It will mount a volume for any local files (e.g. prompt templates, logs).
This service will connect to:
Zep API (probably http://zep:someport, if Zep exposes an API; in Zep CE, maybe it’s integrated with Graphiti, or it might serve on a separate UI port. We’ll clarify ports in Deployment.)
Qdrant (http://qdrant:6333 or via Qdrant client library).
Neo4j (bolt://neo4j:7687 via Neo4j Python driver).
Firecrawl (http://firecrawl-api:3002).
MCP server (if separate, or orchestrator might itself host MCP endpoints).
We might combine MCP server and orchestrator in one process for simplicity. E.g., using the MCP Python SDK, we can define our tools and directly call our orchestrator’s functions for them. Then launch the MCP server within the orchestrator container. This way, if Claude Desktop connects, it can use the tools as well, and if not, it doesn’t harm the orchestrator’s own operation.
LangGraph Implementation: We will use the LangChain/LangGraph Python library inside the orchestrator:
Define memory using either ZepChatMessageHistory (LangChain has integration for Zep) or a custom long-term memory class.
Define retrievers: a VectorStoreRetriever for Qdrant and perhaps a custom retriever that wraps Neo4j queries (LangChain might have Neo4jGraph support for querying).
Define tools as LangChain tools if letting the agent use them (LangChain has a concept of Tools with name, func, description).
Construct a GraphChain or Workflow that sequences these.
Alternatively, write procedural code using LangGraph primitives: e.g.
with Workflow() as wf:
    memory = FetchMemory(input)
    graph_ctx = QueryGraph(input)
    docs = SearchVectorDB(input)
    context = Combine(memory, graph_ctx, docs)
    answer = LLMAnswer(input + context)
    StoreMemory(input, answer)
    Output(answer)
(Pseudo-code to illustrate.)
LangGraph can manage branches: e.g., if graph_ctx is empty and docs are empty, then branch to WebSearch node, else skip it.
Moderation & Safety: Because the system is self-hosted, we must implement any content moderation if needed. Claude has built-in safety to some extent, but we could also intercept at orchestrator level:
For example, before using a tool, ensure the query is not instructing browsing something disallowed. Firecrawl itself obeys robots.txt and has anti-abuse, but on a closed network likely fine.
We can also log all tool usage for auditing via observability tools (next section).
Data Path Summary:
Memory data: stored in Zep’s Postgres (if it uses one internally) or in Neo4j via Graphiti. Zep container likely includes its own DB or uses files; anyway, persists on disk via volume.
Vector data: persisted in Qdrant volume (we specify one).
Graph data: persisted in Neo4j volume.
Ingestion outputs: ephemeral in pipeline but ultimately land in the above stores.
Inter-service communication: All via Docker network (we will use a network alias so containers can talk by name).
We ensure to define ports for external access where needed:
Claude’s MCP connection: If using Claude Desktop, it connects from host to MCP server on Jarvis. We’ll expose the MCP port (e.g. 5100) to the host.
Firecrawl’s port 3002 could be exposed if we want to manually trigger it from host for debugging (not strictly required for Claude’s usage, but we might expose for convenience).
Neo4j’s 7474 we might expose to use Neo4j Browser interface from a web browser for dev/admin.
Qdrant’s port maybe expose if we want to inspect via REST from host.
Others like Zep’s UI or VNC, see next section.
5. Observability and Evaluation
To monitor the agent’s behavior and continuously improve it, we integrate observability tools:
LangSmith (LangChain Tracking): We can enable LangChain’s CallbackManager with LangSmith to log all prompts, model responses, tool usages, latencies, etc. However, note that self-hosting LangSmith is an enterprise feature (the open-source is not fully available). As an alternative, we might use the LangSmith cloud (not fully local) or skip LangSmith and use open tools as below. Another alternative is LangFuse, an open-source competitor that can self-host similar logging (we mention it as an aside). For now:
We will instrument the agent with OpenTelemetry events for each step. This allows feeding data to Arize Phoenix (which supports OpenTelemetry traces).
Arize Phoenix: Phoenix is an open-source LLM observability and evaluation platform. We include it to monitor and evaluate our agent’s performance:
Phoenix provides tracing, capturing each request/response along with metadata. By using Phoenix’s Python integration (arize-phoenix package), we can log each conversation turn, each tool invocation, and their outcome.
It also supports LLM evaluation, where one model can grade the responses of another for relevance or correctness. We may use this to periodically evaluate answers (though doing it with Claude itself offline might be tricky; Phoenix can use a smaller model or just provide evaluation sets for manual review).
Deployment: We can run Phoenix in a container (arizephoenix/phoenix image). Phoenix’s UI typically runs on a port (say 8080). We will expose it at localhost:8085 (for example) for access via browser. The Phoenix backend can use local files or an in-memory DB to store traces, or connect to their cloud if configured (we’ll keep it local).
The orchestrator will push data to Phoenix either via the Python API (client mode) or via OpenTelemetry traces that Phoenix collects. For simplicity, using the Python client is fine: after each conversation, call phoenix.log_trace(conversation_trace) etc., and we can later open the Phoenix web UI to inspect all interactions, chain of thought, and retrieval relevance.
Phoenix is designed to be framework-agnostic and supports LangChain out-of-the-box. We should leverage that by possibly wrapping our LangGraph calls in LangChain’s trace contexts (which Phoenix can capture).
Weave (W&B Weave): Weave is another toolkit (by Weights & Biases) for LLM app monitoring. It can track inputs/outputs and provide a dashboard. However, W&B Weave typically logs to the W&B cloud by default (though an open-source local deployment might be possible; W&B tends to require login). If strictly no cloud, we might skip Weave or run an on-prem W&B server (which is non-trivial). Instead, we focus on Phoenix, but design the system such that adding Weave later is easy (both are just additional instrumentation).
For completeness: Weave can be integrated by adding a few lines in code (decorators or context managers) to log model calls, similar to LangSmith. If desired, the user could sign in to W&B on Jarvis and have the logs stored in their (cloud or local) W&B instance. We won’t do this by default due to “no cloud” rule.
Logging and Analytics: The Zep UI (if running) provides some analytics about memory usage and knowledge graph state. We will expose it (Zep uses a VNC web interface possibly, see below).
Additionally, we might maintain a simple log file (or database) of all user queries and answers on the filesystem for backup.
We ensure all container logs (stdout/err) are captured (Docker will do so; we might mount a logs/ volume for orchestrator to write custom logs or transcripts).
Evaluation datasets: Over time, one can feed known Q&A pairs into the system and use Phoenix’s evaluation or simple diffing to see if answers improve. The system supports creating datasets of test queries in Phoenix which can be run through the agent periodically to detect regressions.
Observability Deployment: We will include an optional Phoenix service in the compose. Alternatively, since Phoenix can be run as a one-off analysis (like Jupyter mode), we might not keep it always running. But we’ll show it in compose for completeness:
The Phoenix container will likely incorporate a web UI accessible at http://<server>:8080 (we’ll map it to 8085 externally to avoid clash with any other service like perhaps Zep).
Phoenix’s container can store its data in a volume (if it has a database; it might just run in-memory unless configured with persistent storage).
We can configure Phoenix via environment variables if needed (e.g. to point it to traces or to run in offline mode with no cloud). The default pip usage just allows saving and loading trace data from files or not needed if we keep it live.
Summary: With Phoenix (and/or Weave) integrated, we will have:
Real-time view of agent’s internal steps (the chain from memory fetch to final answer) via trace UI.
Ability to evaluate answers against ground truth (if we supply some) or via LLM graders for quality.
System metrics like token usage, latency of each tool, etc., to identify bottlenecks (maybe Firecrawl calls are slow, etc., then we know to adjust).
This closes the loop, enabling continuous improvement of our Jarvis AI system in a self-hosted manner.
6. Deployment Setup and Configuration
We now outline the concrete setup instructions, including Docker Compose definitions, environment configuration, and default data paths/ports for each component. The goal is to provide an implementation-ready guide that Claude Code (or any operator) can follow to launch the system on the Jarvis server.
6.1 System Requirements
Server: The Jarvis server should have sufficient resources (recommend at least 16 GB RAM, 4+ CPU cores, and 100+ GB of disk) especially if running heavy components like a browser or large databases. If running a local LLM model for Claude, ensure a GPU or enough CPU and RAM for inference.
OS: Linux (Docker Engine installed). Docker Compose v2 (or Docker Compose plugin) is required.
Network: No external dependencies required except optional Brave API or Claude API. However, internet access is needed initially to pull Docker images and to crawl documentation (Firecrawl will perform HTTP requests to target sites).
6.2 Directory Layout and Data Paths
We will use a base directory, e.g. /opt/jarvis_ai/ on the server, to organize files:
docker-compose.yml (main orchestration file for all core services).
.env files or config for specific services (if needed).
firecrawl/ directory containing the cloned Firecrawl repo (if building from source).
data/ directory to mount persistent volumes (for Neo4j, Qdrant, etc.):
data/neo4j/ – Neo4j database files.
data/qdrant/ – Qdrant storage.
data/zep/ – (If needed, e.g. if Zep writes any logs or DB, depending on image).
(Redis data we keep ephemeral unless needed; same for Firecrawl ephemeral caches.)
logs/ directory for orchestrator logs or output (optional).
All persistent volumes in Docker Compose will be mapped under ./data for clarity.
6.3 Environment Variables and Secrets
We will define necessary environment variables in a .env file loaded by compose. Key settings:
NEO4J_AUTH=user/password and possibly NEO4J_USER and NEO4J_PASSWORD (for Graphiti).
OPENAI_API_KEY for Graphiti (if using OpenAI for graph operations). If we want to avoid external, we might provide a dummy or, better, configure Graphiti to use a local LLM – Graphiti currently doesn’t support local LLM out-of-box, so we give it an API key to at least a small model (like OpenAI gpt-3.5) for its operations. This is a slight cloud dependency; if unacceptable, we can disable those features of Graphiti (it may still build the graph from provided data without needing to call the LLM).
ZEP_OPENAI_API_KEY (if Zep needs it separately; possibly it uses Graphiti’s key).
CLAUDE_API_KEY if using Claude’s API (if Claude Code is accessible via API and not purely through desktop+MCP). Alternatively, if using Claude locally via desktop MCP, no key needed but we need to ensure Claude Desktop is installed and connected to our MCP.
BRAVE_API_KEY if Brave search is used.
FIRECRAWL_NO_AUTH=true (or similar) to indicate self-host mode with no API key needed (since we disabled DB auth).
Any other optional service keys (ScrapingBee proxy for Firecrawl if needed, etc.) left blank by default.
We store sensitive keys in .env which is not committed anywhere, just on the server.
6.4 Docker Compose Configuration
Below is the Docker Compose configuration for the core system. Each service is defined with necessary settings (image, ports, volumes, environment):
version: '3.8'
services:
  # Vector Database (Qdrant)
  qdrant:
    image: qdrant/qdrant:v1.4.0   # example version
    container_name: jarvis_qdrant
    ports:
      - "6333:6333"    # HTTP API
      - "6334:6334"    # gRPC (if needed)
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped

  # Graph Database (Neo4j)
  neo4j:
    image: neo4j:5.12.0-community
    container_name: jarvis_neo4j
    ports:
      - "7474:7474"    # HTTP browser UI (optional)
      - "7687:7687"    # Bolt protocol for DB
    volumes:
      - ./data/neo4j:/data
    environment:
      - NEO4J_AUTH=${NEO4J_AUTH}   # e.g. "neo4j/secretpassword"
    restart: unless-stopped

  # Graph Memory / GraphRAG Service (Graphiti)
  graphiti:
    image: zepai/graphiti:0.3      # Graphiti server (FastAPI)
    container_name: jarvis_graphiti
    ports:
      - "8000:8000"    # Graphiti HTTP API (internal use, can be closed externally)
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=${NEO4J_USER:-neo4j}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      - neo4j
    restart: unless-stopped

  # Long-term Memory Service (Zep)
  zep:
    image: zepai/zep:1.0
    container_name: jarvis_zep
    # Zep might have a UI via VNC; expose if needed.
    ports:
      - "8080:8080"        # assume Zep UI or API
      - "5901:5901"        # VNC port if using KasmVNC
      - "6901:6901"        # Web VNC port
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ZEP_GRAPHITI_URL=http://graphiti:8000   # (If Zep needs to know Graphiti endpoint)
      # Additional env vars could be set if required by Zep
    depends_on:
      - graphiti
    restart: unless-stopped

  # Redis (for Firecrawl)
  redis:
    image: redis:7-alpine
    container_name: jarvis_redis
    command: ["redis-server", "--save", "", "--appendonly", "no"]  # no persistence needed
    ports:
      - "6379:6379"   # not exposed externally in production, but here for completeness
    restart: unless-stopped

  # Playwright browser service (for Firecrawl, Python version)
  playwright-service:
    build: 
      context: ./firecrawl/apps/playwright-service
      dockerfile: Dockerfile
    container_name: jarvis_playwright
    ports:
      - "3000:3000"   # not exposed externally, internal to Firecrawl
    restart: unless-stopped

  # Firecrawl API service
  firecrawl-api:
    build:
      context: ./firecrawl/apps/api
      dockerfile: Dockerfile
    container_name: jarvis_firecrawl
    env_file: ./firecrawl/.env    # configuration as per Firecrawl's template
    ports:
      - "3002:3002"
    depends_on:
      - redis
      - playwright-service
    restart: unless-stopped

  # Orchestrator & MCP Server (Jarvis Agent Orchestrator)
  orchestrator:
    build: 
      context: ./orchestrator    # assumes we have a Dockerfile to set up LangChain, etc.
      dockerfile: Dockerfile
    container_name: jarvis_orchestrator
    # We assume orchestrator code will start the MCP server internally on 5100
    ports:
      - "5100:5100"   # MCP server port for Claude to connect
    volumes:
      - ./logs:/app/logs        # mount for logs (optional)
      - ./data:/app/data:ro     # read-only access to data if needed (not strictly necessary)
    environment:
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      # (plus any other config orchestrator needs, like endpoints or flags)
    depends_on:
      - qdrant
      - neo4j
      - graphiti
      - zep
      - firecrawl-api
    restart: unless-stopped

  # Observability - Arize Phoenix (optional; can disable if not needed)
  phoenix:
    image: arizephoenix/phoenix:latest
    container_name: jarvis_phoenix
    ports:
      - "8085:8080"   # Phoenix UI (mapped to 8085 on host)
    environment:
      - PHOENIX_API_KEY=   # no key needed for local usage
      # (Phoenix might not need many env vars; it runs an API/UI for logged traces)
    volumes:
      - ./data/phoenix:/phoenix/data   # if Phoenix stores data
    restart: unless-stopped
(Note: The build contexts for Firecrawl require that you have the Firecrawl repository present at ./firecrawl. You should clone the repository and check out a stable release tag. Alternatively, if Mendable provides pre-built images in the future, you can replace build with image: mendableai/firecrawl-api:latest and similarly for playwright. As of writing, self-host requires building from source.) A few points about this configuration:
We used fixed container names (prefixed with jarvis_) for clarity, but it’s optional.
We mapped host ports for services that we may interact with during development:
Qdrant (6333) if we want to manually query it or use its UI.
Neo4j (7474) so we can open Neo4j Browser at http://localhost:7474 (and log in with the credentials we set) to visually inspect the knowledge graph.
Firecrawl API (3002) if we want to trigger crawls manually via curl or Postman from host.
MCP orchestrator (5100) because Claude (if using Claude Desktop) will connect from the host to this port. In production, if Claude runs on same machine or if orchestrator itself calls Claude’s API, this might not be needed.
Phoenix UI (8085 mapped to container 8080).
Zep’s ports: This is speculative; the Zep container might expose a web interface on 8080 (with a KasmVNC environment to show analytics). We exposed 5901/6901 which are typical VNC ports for Kasm images. After starting, one can try connecting to http://localhost:6901 to see if a GUI appears. This interface could show Zep’s analytics dashboard (if implemented). If not needed, these can be removed or changed. Zep might also have a direct REST API on some port, but documentation is sparse – if it uses Graphiti and the Graphiti port, perhaps not.
The orchestrator is built from a local context ./orchestrator. We need to create this:
The Dockerfile would set up a Python environment with needed packages: langchain, langchain-ai (for LangGraph), anthropic (Claude API), neo4j, qdrant-client, arize-phoenix (for logging), etc.
It would copy our orchestrator code (a Python script or package) and run it (maybe via CMD ["python", "-m", "jarvis_orchestrator"]).
The orchestrator on start should initialize the MCP server (listening on 0.0.0.0:5100) and register tools. It can also immediately begin a loop to listen for requests (if using Claude via MCP). If using Claude API, it might instead expose a simple web or CLI interface for user queries (but since user likely uses Claude as interface, we focus on MCP route).
In our design, we assume the user will chat with Claude through the Claude app or interface, and the Claude -> MCP -> orchestrator pathway handles queries. We could also have orchestrator open a local chat UI or CLI as an alternative interface.
After creating the above file, run the deployment with:
# Navigate to the base directory
cd /opt/jarvis_ai

# Ensure the Firecrawl repository is present for building images
git clone https://github.com/mendableai/firecrawl.git firecrawl
# (Optional) checkout a specific release known to be stable
# git checkout tags/v1.15.0  (if such tag exists, for example)

# Copy environment template for Firecrawl
cp firecrawl/apps/api/.env.example firecrawl/.env
# Edit firecrawl/.env to set at least:
#   NUM_WORKERS_PER_QUEUE=4 (for example, to reduce load)
#   HOST=0.0.0.0, PORT=3002 (already default)
#   REDIS_URL=redis://redis:6379 (for docker setup)
#   PLAYWRIGHT_MICROSERVICE_URL=http://playwright-service:3000/html
#   USE_DB_AUTHENTICATION=false, and all API keys blank for now.

# Create orchestrator code and Dockerfile (as per Section 6.5 below).

# Launch all services
docker-compose up -d
This will pull/build images and start containers in the background. Use docker-compose logs -f orchestrator to monitor the orchestrator startup, and ensure each service comes up without errors:
Neo4j will print logs indicating it’s ready and Bolt connector is listening.
Graphiti will log that it’s connected to Neo4j (and maybe waiting for queries).
Zep might take longer to start; look for any log that it’s ready or listening.
Firecrawl will log the API endpoint (you should see something like “Server listening on port 3002”).
Orchestrator will likely log the MCP server startup (e.g. “MCP server listening at 0.0.0.0:5100”).
Phoenix will log a message about its web interface being available.
Check that:
Qdrant’s storage directory ./data/qdrant is created.
Neo4j’s data directory has files (and you can open http://<server>:7474 in a browser, it should ask for user/pass which you set as env).
Firecrawl is accessible: try curl http://localhost:3002/v0/health (if such exists) or check docker-compose logs firecrawl-api for something like “Listening on 3002”.
The orchestrator doesn’t crash. If it’s waiting for connections (Claude via MCP), it might just sit quietly after initialization.
6.5 Orchestrator Implementation Notes
In the orchestrator code (which is beyond the scope of this text to fully write out), we will do roughly:
Initialize LangChain/LangGraph components:
Load our API keys from env (for Claude, etc.).
Create a ConversationBufferMemory if short-term needed (though we lean on long-term memory).
Create a ZepMemory object (LangChain provides one) if using Zep, configured with Zep’s base URL (likely http://zep:8000 or another port if Zep has an API).
Initialize Qdrant client: e.g. qdrant_client = QdrantClient(url="http://qdrant:6333").
Initialize Neo4j driver: e.g. using NEO4J_URI, NEO4J_USER/PASS from env.
Define a retrieval function for docs: e.g. vector_retriever = QdrantRetriever(qdrant_client, collection_name="docs", embedding_model=embedding_fn).
Define graph query function: can be direct Cypher or using Graphiti’s REST. For simplicity, maybe use Graphiti REST by calling requests.get("http://graphiti:8000/query", params={...}) if such exists. If not, directly use neo4j_driver.session().run(<cypher>).
Set up MCP server via modelcontext library:
e.g. server = MCPServer(host="0.0.0.0", port=5100)
Register tools:
@server.tool(name="SearchDocs", description="...") def search_docs(query: str): ... that uses vector_retriever and returns text.
Similarly @server.tool(name="GetGraphInfo", ...) def get_graph(entity: str): ... that queries Neo4j for that entity’s neighbors and returns a summary.
@server.tool(name="WebSearch", ...) def web_search(query: str): ... that calls Firecrawl’s search and returns result titles.
@server.tool(name="WebScrape", ...) def web_scrape(url: str): ... that calls Firecrawl’s scrape and returns text.
Start the server: server.run() or similar.
MCP Interaction: If using Claude via the desktop app, the app will handle sending the conversation to MCP. We ensure our tools produce concise JSON outputs or text that Claude can parse.
If using Claude API programmatically (instead of the desktop UI), we would have to implement the loop of sending Claude a prompt, getting output, and checking for tool invocation format (MCP spec uses a specific message structure). This is more complex and likely not needed if we use Claude’s built-in support via the desktop. So we assume the primary interface is Claude with MCP.
Usage Note: To use the system, one would:
Open Claude (Desktop) on their machine, connect it to the MCP server (http://<jarvis_server>:5100 as per Anthropic’s guide).
Then in Claude’s chat, ask questions. Claude will then behind the scenes call MCP tools. For example:
User asks: “What does the API endpoint /v1/orders do in Acme’s service? And was there any change in 2023?”
Claude will perhaps call SearchDocs with query "API /v1/orders Acme 2023 change" via MCP.
Our orchestrator executes that: Qdrant returns a doc snippet saying “/v1/orders: create new order. Changed in v2.0 (2023) – requires an API key.”
Claude gets that result, then maybe it calls GetGraphInfo with entity "Acme API v1/orders". The graph might have a node for that endpoint linked to a "version 2.0 (2023)" node.
Orchestrator returns: “/v1/orders --updated-> 2023-05-01 (version 2.0 release)” for example.
Claude now has enough and formulates an answer about the API and its change in 2023.
Finally, orchestrator logs all this to Phoenix; we can later see the tool calls timeline and the answer.
6.6 Persistent Storage and Backup
The system’s knowledge and state reside in:
Neo4j graph (persisted under ./data/neo4j).
Qdrant vector store (./data/qdrant).
Zep memory (possibly in an internal SQLite or files in its container – if so, mount ./data/zep if needed; we might adjust the compose to volume-mount a data dir if we find where Zep stores community edition data).
Logs in ./logs and Phoenix data in ./data/phoenix (if used).
Regular backups of these volumes are recommended. Neo4j can be backed up via its dump utilities or by copying the database files when the container is stopped. Qdrant data can be copied as well (it’s usually just a directory of segment files). Since everything is local, one could also snapshot the entire data/ folder.
6.7 Security and Access
Because everything is on a single server, firewall the ports as needed:
Expose MCP (5100) only to the users who need to connect (if server is within a LAN, ensure it’s not open to internet).
Similarly, Firecrawl (3002) and Neo4j UI (7474) could be sensitive; limit access via firewall or Docker’s network settings (we could remove the ports: mapping for 3002 and 7474 if not needed externally – the orchestrator and Claude (via MCP server) can still reach them on the internal network).
If multiple people use the system, consider enabling authentication:
Firecrawl supports API keys (via Supabase integration, though not trivial offline).
MCP server could be behind an SSH tunnel or VPN if needed.
Neo4j has its user/pass; use a strong password there.
Claude’s outputs should be monitored if connected to the internet (in our design, minimal external calls are made).
The server should be secured (since we run a browser via Firecrawl that might fetch untrusted content, ensure OS packages (like Chrome in Playwright) are up-to-date to avoid exploits. Firecrawl’s container isolates that, but still good practice).
6.8 Testing and Usage Notes
Once setup:
Test memory: Ask something that was mentioned in a previous session to see if Zep recalls it.
Test doc retrieval: Ask a question from ingested docs and ensure the answer cites the correct info (if we program Claude to cite sources, which we can – we might instruct Claude to include source names if desired).
Test graph: Ask a question requiring a relationship, e.g. “Which company makes Product X and what other products do they have?” If our graph has that, see if answer comes from graph data.
Check Phoenix: Run a few queries, then go to http://localhost:8085 and explore the traces. You should see events for each tool call and the prompt/response. Phoenix can also highlight any hallucinations if we feed it expected answers for some queries (advanced usage).
The design above aims for modularity: each component (memory, vector DB, graph, etc.) can be replaced or upgraded independently. For instance, if you prefer Weaviate over Qdrant, you can swap the service and adjust the orchestrator’s vector retrieval code. If you want to try a different LLM, you could connect the MCP to an open-source model (e.g. through OpenAI’s function calling interface or LangChain Agents). Finally, ensure to update documentation as needed and monitor resource usage (the Playwright browser can be heavy; in low-memory environment consider reducing NUM_WORKERS_PER_QUEUE to 1 or 2 so only one headless browser instance runs at a time).
Sources:
LangMem introduction (long-term memory for LLMs)\
Zep memory platform description\
Zep Community Edition (graph-based memory) announcement\
GraphRAG concept and benefits of hybrid approach\
Apify blog on Firecrawl vs Tavily (web ingestion approaches)\
Firecrawl architecture (crawling, headless browser)\
LangGraph orchestration benefits (control flows, memory)\
Model Context Protocol intro (standardizing AI-tool integration)\
Arize Phoenix features (tracing and evaluation for LLM apps)\
W&B Weave purpose (LLM app monitoring toolkit)
Bronvermeldingen

GraphRAG with Qdrant and Neo4j - Qdrant

https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/
GitHub
https://github.com/kindship-ai/firecrawl-cli/blob/a9a3fed5d325560183f3d4cbd22ace1fd77aa955/docs_firecrawl_dev/contributing_self-host_self-hosting-firecrawl.md#L240-L248
GitHub
https://github.com/kindship-ai/firecrawl-cli/blob/a9a3fed5d325560183f3d4cbd22ace1fd77aa955/docs_firecrawl_dev/contributing_self-host_self-hosting-firecrawl.md#L63-L71
GitHub
https://github.com/kindship-ai/firecrawl-cli/blob/a9a3fed5d325560183f3d4cbd22ace1fd77aa955/docs_firecrawl_dev/contributing_self-host_self-hosting-firecrawl.md#L86-L94
GitHub
https://github.com/kindship-ai/firecrawl-cli/blob/a9a3fed5d325560183f3d4cbd22ace1fd77aa955/docs_firecrawl_dev/contributing_self-host_self-hosting-firecrawl.md#L74-L82
Alle bronnen

qdrant

github
