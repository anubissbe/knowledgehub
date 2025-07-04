# Architecture Diagrams and System Flow

This document provides visual representations of the KnowledgeHub system architecture, data flows, and component interactions.

## 📊 System Overview Diagram

```mermaid
graph TB
    subgraph "External Sources"
        WEB[📄 Documentation Sites]
        GIT[🔧 GitHub Repositories]
        FILE[📁 File Uploads]
    end
    
    subgraph "KnowledgeHub Platform"
        subgraph "Web Layer"
            UI[🖥️ React Frontend]
            API[🔌 FastAPI Backend]
        end
        
        subgraph "Processing Layer"
            SCRAPER[🕷️ Web Scraper]
            RAG[🤖 RAG Processor]
            SCHEDULER[⏰ Job Scheduler]
        end
        
        subgraph "Storage Layer"
            POSTGRES[🐘 PostgreSQL]
            REDIS[⚡ Redis Cache]
            WEAVIATE[🧠 Vector DB]
            MINIO[📦 Object Storage]
        end
    end
    
    subgraph "Infrastructure"
        NGINX[🌐 Nginx Proxy]
        DOCKER[🐳 Docker Containers]
        GPU[⚡ GPU Acceleration]
    end
    
    %% Data Flow
    WEB --> SCRAPER
    GIT --> SCRAPER
    FILE --> API
    
    UI --> API
    API --> POSTGRES
    API --> REDIS
    API --> WEAVIATE
    
    SCRAPER --> RAG
    RAG --> POSTGRES
    RAG --> WEAVIATE
    RAG --> MINIO
    
    SCHEDULER --> SCRAPER
    REDIS --> SCHEDULER
    
    NGINX --> UI
    NGINX --> API
    
    %% Styling
    classDef external fill:#e1f5fe
    classDef web fill:#f3e5f5
    classDef processing fill:#fff3e0
    classDef storage fill:#e8f5e8
    classDef infra fill:#fce4ec
    
    class WEB,GIT,FILE external
    class UI,API web
    class SCRAPER,RAG,SCHEDULER processing
    class POSTGRES,REDIS,WEAVIATE,MINIO storage
    class NGINX,DOCKER,GPU infra
```

## 🔄 Data Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant Scraper
    participant RAG
    participant VectorDB
    participant Database
    participant Cache
    
    User->>Frontend: Create Knowledge Source
    Frontend->>API: POST /sources
    API->>Database: Store source metadata
    API->>Scraper: Queue crawl job
    
    Note over Scraper: Incremental Crawling
    Scraper->>Scraper: Load existing docs cache
    Scraper->>API: Get existing documents
    loop For each URL
        Scraper->>Scraper: Calculate content hash
        alt Content unchanged
            Scraper->>Scraper: Skip processing
        else Content new/changed
            Scraper->>RAG: Send for processing
        end
    end
    
    Note over RAG: Content Processing
    RAG->>RAG: Clean and chunk content
    RAG->>RAG: Generate embeddings (GPU)
    RAG->>Database: Store document chunks
    RAG->>VectorDB: Store embeddings
    RAG->>API: Update job status
    
    Note over User: Search Process
    User->>Frontend: Search query
    Frontend->>API: POST /search
    API->>Cache: Check cache
    alt Cache hit
        Cache->>API: Return cached results
    else Cache miss
        par Parallel Search
            API->>VectorDB: Vector similarity search
        and
            API->>Database: Keyword search
        end
        API->>API: Merge and rank results
        API->>Cache: Cache results
    end
    API->>Frontend: Return search results
    Frontend->>User: Display results
```

## 🏗️ Component Architecture

```mermaid
graph LR
    subgraph "Frontend (React)"
        COMP[Components]
        HOOKS[Hooks]
        STORE[State Management]
        ROUTING[Routing]
    end
    
    subgraph "Backend (FastAPI)"
        ROUTES[API Routes]
        SERVICES[Business Logic]
        MODELS[Data Models]
        DEPS[Dependencies]
    end
    
    subgraph "Workers"
        CRAWL[Crawler Service]
        PROCESS[RAG Processor]
        EMBED[Embeddings Service]
        SCHED[Scheduler Service]
    end
    
    subgraph "Databases"
        PG[(PostgreSQL)]
        REDIS[(Redis)]
        WEAVIATE[(Weaviate)]
        MINIO[(MinIO)]
    end
    
    %% Frontend connections
    COMP --> HOOKS
    HOOKS --> STORE
    STORE --> ROUTING
    
    %% API connections
    ROUTES --> SERVICES
    SERVICES --> MODELS
    MODELS --> DEPS
    
    %% Cross-layer connections
    STORE -.->|HTTP/WebSocket| ROUTES
    SERVICES --> PG
    SERVICES --> REDIS
    SERVICES --> WEAVIATE
    
    %% Worker connections
    CRAWL --> PROCESS
    PROCESS --> EMBED
    SCHED --> CRAWL
    
    PROCESS --> PG
    PROCESS --> WEAVIATE
    EMBED --> MINIO
```

## 🔍 Search Architecture

```mermaid
graph TD
    subgraph "Search Request Flow"
        QUERY[User Query]
        CACHE{Cache Check}
        SEARCH[Search Service]
        
        subgraph "Search Types"
            HYBRID[Hybrid Search]
            VECTOR[Vector Search]
            KEYWORD[Keyword Search]
        end
        
        subgraph "Data Sources"
            VSTORE[Vector Store]
            FULLTEXT[Full-Text Index]
        end
        
        MERGE[Result Merging]
        RANK[Ranking Algorithm]
        RESULTS[Search Results]
    end
    
    QUERY --> CACHE
    CACHE -->|Miss| SEARCH
    CACHE -->|Hit| RESULTS
    
    SEARCH --> HYBRID
    SEARCH --> VECTOR
    SEARCH --> KEYWORD
    
    VECTOR --> VSTORE
    KEYWORD --> FULLTEXT
    HYBRID --> VSTORE
    HYBRID --> FULLTEXT
    
    VSTORE --> MERGE
    FULLTEXT --> MERGE
    MERGE --> RANK
    RANK --> RESULTS
    
    %% Styling
    classDef query fill:#e3f2fd
    classDef process fill:#f1f8e9
    classDef data fill:#fff3e0
    classDef result fill:#fce4ec
    
    class QUERY query
    class CACHE,SEARCH,HYBRID,VECTOR,KEYWORD,MERGE,RANK process
    class VSTORE,FULLTEXT data
    class RESULTS result
```

## 📱 Frontend Component Hierarchy

```mermaid
graph TD
    APP[App]
    
    subgraph "Layout"
        LAYOUT[Layout]
        NAV[Navigation]
        SIDEBAR[Sidebar]
        MAIN[Main Content]
    end
    
    subgraph "Pages"
        DASH[Dashboard]
        SOURCES[Sources]
        SEARCH[Search]
        JOBS[Jobs]
        SETTINGS[Settings]
    end
    
    subgraph "Components"
        CARDS[Source Cards]
        TABLES[Data Tables]
        FORMS[Forms]
        CHARTS[Charts]
        MODAL[Modals]
    end
    
    subgraph "Hooks"
        API_HOOKS[API Hooks]
        STATE_HOOKS[State Hooks]
        UTIL_HOOKS[Utility Hooks]
    end
    
    APP --> LAYOUT
    LAYOUT --> NAV
    LAYOUT --> SIDEBAR
    LAYOUT --> MAIN
    
    MAIN --> DASH
    MAIN --> SOURCES
    MAIN --> SEARCH
    MAIN --> JOBS
    MAIN --> SETTINGS
    
    DASH --> CARDS
    DASH --> CHARTS
    SOURCES --> TABLES
    SOURCES --> FORMS
    SEARCH --> TABLES
    JOBS --> TABLES
    
    CARDS --> API_HOOKS
    TABLES --> API_HOOKS
    FORMS --> STATE_HOOKS
    CHARTS --> UTIL_HOOKS
    
    %% Styling
    classDef app fill:#e8eaf6
    classDef layout fill:#f3e5f5
    classDef pages fill:#e0f2f1
    classDef components fill:#fff3e0
    classDef hooks fill:#fce4ec
    
    class APP app
    class LAYOUT,NAV,SIDEBAR,MAIN layout
    class DASH,SOURCES,SEARCH,JOBS,SETTINGS pages
    class CARDS,TABLES,FORMS,CHARTS,MODAL components
    class API_HOOKS,STATE_HOOKS,UTIL_HOOKS hooks
```

## 🔧 Processing Pipeline

```mermaid
flowchart TD
    START([Source Added])
    
    subgraph "Crawling Phase"
        INIT[Initialize Crawler]
        CACHE[Load Document Cache]
        CRAWL[Crawl Pages]
        HASH[Calculate Content Hash]
        CHANGED{Content Changed?}
        EXTRACT[Extract Content]
        SKIP[Skip Processing]
    end
    
    subgraph "Processing Phase"
        CLEAN[Clean Content]
        CHUNK[Create Chunks]
        EMBED[Generate Embeddings]
        STORE[Store in Database]
        INDEX[Index in Vector DB]
    end
    
    subgraph "Completion Phase"
        UPDATE[Update Statistics]
        NOTIFY[Send Notifications]
        CACHE_UPDATE[Update Cache]
        COMPLETE([Job Complete])
    end
    
    START --> INIT
    INIT --> CACHE
    CACHE --> CRAWL
    CRAWL --> HASH
    HASH --> CHANGED
    
    CHANGED -->|Yes| EXTRACT
    CHANGED -->|No| SKIP
    SKIP --> CRAWL
    
    EXTRACT --> CLEAN
    CLEAN --> CHUNK
    CHUNK --> EMBED
    EMBED --> STORE
    STORE --> INDEX
    
    INDEX --> UPDATE
    UPDATE --> NOTIFY
    NOTIFY --> CACHE_UPDATE
    CACHE_UPDATE --> COMPLETE
    
    %% Styling
    classDef start fill:#c8e6c9
    classDef crawl fill:#bbdefb
    classDef process fill:#ffecb3
    classDef complete fill:#f8bbd9
    classDef decision fill:#ffcdd2
    
    class START,COMPLETE start
    class INIT,CACHE,CRAWL,HASH,EXTRACT,SKIP crawl
    class CLEAN,CHUNK,EMBED,STORE,INDEX process
    class UPDATE,NOTIFY,CACHE_UPDATE complete
    class CHANGED decision
```

## 🚀 Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        subgraph "Load Balancer"
            LB[Nginx Load Balancer]
            SSL[SSL Termination]
        end
        
        subgraph "Application Tier"
            subgraph "Web Services"
                API1[API Instance 1]
                API2[API Instance 2]
                UI1[Frontend Instance 1]
                UI2[Frontend Instance 2]
            end
            
            subgraph "Worker Services"
                SCRAPER1[Scraper Worker 1]
                SCRAPER2[Scraper Worker 2]
                RAG1[RAG Processor 1]
                RAG2[RAG Processor 2]
                EMBED1[Embeddings Service]
                SCHED1[Scheduler Service]
            end
        end
        
        subgraph "Data Tier"
            subgraph "Primary Storage"
                PG_MASTER[(PostgreSQL Master)]
                PG_REPLICA[(PostgreSQL Replica)]
            end
            
            subgraph "Cache & Vector"
                REDIS_CLUSTER[(Redis Cluster)]
                WEAVIATE_CLUSTER[(Weaviate Cluster)]
            end
            
            subgraph "Object Storage"
                MINIO_CLUSTER[(MinIO Cluster)]
            end
        end
        
        subgraph "Infrastructure"
            subgraph "Container Orchestration"
                DOCKER[Docker Swarm / K8s]
                MONITOR[Monitoring Stack]
            end
            
            subgraph "GPU Resources"
                GPU1[GPU Node 1]
                GPU2[GPU Node 2]
            end
        end
    end
    
    %% Load Balancer connections
    LB --> API1
    LB --> API2
    LB --> UI1
    LB --> UI2
    
    %% Database connections
    API1 --> PG_MASTER
    API2 --> PG_MASTER
    API1 --> PG_REPLICA
    API2 --> PG_REPLICA
    
    %% Cache connections
    API1 --> REDIS_CLUSTER
    API2 --> REDIS_CLUSTER
    
    %% Vector database connections
    API1 --> WEAVIATE_CLUSTER
    API2 --> WEAVIATE_CLUSTER
    
    %% Worker connections
    SCRAPER1 --> RAG1
    SCRAPER2 --> RAG2
    RAG1 --> EMBED1
    RAG2 --> EMBED1
    
    %% GPU connections
    EMBED1 --> GPU1
    EMBED1 --> GPU2
    
    %% Storage connections
    RAG1 --> MINIO_CLUSTER
    RAG2 --> MINIO_CLUSTER
    
    %% Monitoring
    MONITOR --> API1
    MONITOR --> API2
    MONITOR --> SCRAPER1
    MONITOR --> SCRAPER2
    
    %% Styling
    classDef lb fill:#e1f5fe
    classDef web fill:#f3e5f5
    classDef worker fill:#fff3e0
    classDef storage fill:#e8f5e8
    classDef infra fill:#fce4ec
    
    class LB,SSL lb
    class API1,API2,UI1,UI2 web
    class SCRAPER1,SCRAPER2,RAG1,RAG2,EMBED1,SCHED1 worker
    class PG_MASTER,PG_REPLICA,REDIS_CLUSTER,WEAVIATE_CLUSTER,MINIO_CLUSTER storage
    class DOCKER,MONITOR,GPU1,GPU2 infra
```

## 🔐 Security Architecture

```mermaid
graph LR
    subgraph "External Access"
        USER[Users]
        API_CLIENT[API Clients]
    end
    
    subgraph "Security Layer"
        WAF[Web Application Firewall]
        RATE_LIMIT[Rate Limiting]
        AUTH[Authentication]
        AUTHZ[Authorization]
        ENCRYPT[Encryption at Rest]
        TLS[TLS/SSL]
    end
    
    subgraph "Application Layer"
        API[API Gateway]
        SERVICES[Microservices]
        VALIDATION[Input Validation]
        SANITIZATION[Data Sanitization]
    end
    
    subgraph "Data Layer"
        SECRETS[Secret Management]
        AUDIT[Audit Logging]
        BACKUP[Encrypted Backups]
        ACCESS_CONTROL[Access Controls]
    end
    
    USER --> WAF
    API_CLIENT --> WAF
    WAF --> RATE_LIMIT
    RATE_LIMIT --> TLS
    TLS --> AUTH
    AUTH --> AUTHZ
    AUTHZ --> API
    
    API --> VALIDATION
    VALIDATION --> SANITIZATION
    SANITIZATION --> SERVICES
    
    SERVICES --> SECRETS
    SERVICES --> AUDIT
    SERVICES --> ACCESS_CONTROL
    ACCESS_CONTROL --> ENCRYPT
    ENCRYPT --> BACKUP
    
    %% Styling
    classDef external fill:#e3f2fd
    classDef security fill:#fff3e0
    classDef app fill:#f1f8e9
    classDef data fill:#fce4ec
    
    class USER,API_CLIENT external
    class WAF,RATE_LIMIT,AUTH,AUTHZ,ENCRYPT,TLS security
    class API,SERVICES,VALIDATION,SANITIZATION app
    class SECRETS,AUDIT,BACKUP,ACCESS_CONTROL data
```

## 📊 Performance Monitoring

```mermaid
graph TD
    subgraph "Metrics Collection"
        APP_METRICS[Application Metrics]
        SYS_METRICS[System Metrics]
        CUSTOM_METRICS[Custom Metrics]
    end
    
    subgraph "Monitoring Stack"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana Dashboards]
        ALERTMANAGER[Alert Manager]
    end
    
    subgraph "Logging"
        APP_LOGS[Application Logs]
        ACCESS_LOGS[Access Logs]
        ERROR_LOGS[Error Logs]
        ELK[ELK Stack]
    end
    
    subgraph "Alerting"
        SLACK[Slack Notifications]
        EMAIL[Email Alerts]
        WEBHOOK[Webhook Endpoints]
        ONCALL[On-Call System]
    end
    
    subgraph "Analytics"
        USAGE[Usage Analytics]
        PERFORMANCE[Performance Analytics]
        ERROR_TRACKING[Error Tracking]
        REPORTS[Automated Reports]
    end
    
    APP_METRICS --> PROMETHEUS
    SYS_METRICS --> PROMETHEUS
    CUSTOM_METRICS --> PROMETHEUS
    
    PROMETHEUS --> GRAFANA
    PROMETHEUS --> ALERTMANAGER
    
    APP_LOGS --> ELK
    ACCESS_LOGS --> ELK
    ERROR_LOGS --> ELK
    
    ALERTMANAGER --> SLACK
    ALERTMANAGER --> EMAIL
    ALERTMANAGER --> WEBHOOK
    ALERTMANAGER --> ONCALL
    
    GRAFANA --> USAGE
    ELK --> PERFORMANCE
    ELK --> ERROR_TRACKING
    USAGE --> REPORTS
    
    %% Styling
    classDef metrics fill:#e8f5e8
    classDef monitoring fill:#e3f2fd
    classDef logging fill:#fff3e0
    classDef alerting fill:#ffebee
    classDef analytics fill:#f3e5f5
    
    class APP_METRICS,SYS_METRICS,CUSTOM_METRICS metrics
    class PROMETHEUS,GRAFANA,ALERTMANAGER monitoring
    class APP_LOGS,ACCESS_LOGS,ERROR_LOGS,ELK logging
    class SLACK,EMAIL,WEBHOOK,ONCALL alerting
    class USAGE,PERFORMANCE,ERROR_TRACKING,REPORTS analytics
```

## 🔄 Data Synchronization Flow

```mermaid
stateDiagram-v2
    [*] --> Idle
    
    Idle --> ScheduledCheck : Cron Trigger
    Idle --> ManualRefresh : User Request
    Idle --> IncrementalUpdate : Content Change Detected
    
    ScheduledCheck --> LoadCache : Load Existing Documents
    ManualRefresh --> ForceRefresh : Force Full Refresh
    IncrementalUpdate --> LoadCache : Load Existing Documents
    
    LoadCache --> CompareContent : Hash Comparison
    ForceRefresh --> ProcessAll : Process All Pages
    
    CompareContent --> ProcessChanged : Changed Content Found
    CompareContent --> FindNewPages : No Changes, Check for New
    CompareContent --> Complete : No Changes, No New Pages
    
    ProcessChanged --> UpdateEmbeddings : Generate New Embeddings
    ProcessAll --> UpdateEmbeddings : Generate All Embeddings
    FindNewPages --> ProcessNew : New Pages Found
    FindNewPages --> Complete : No New Pages
    
    ProcessNew --> UpdateEmbeddings : Generate Embeddings for New
    
    UpdateEmbeddings --> UpdateDatabase : Store in PostgreSQL
    UpdateDatabase --> UpdateVectorStore : Store in Weaviate
    UpdateVectorStore --> InvalidateCache : Clear Redis Cache
    InvalidateCache --> Complete : Update Statistics
    
    Complete --> Idle : Wait for Next Trigger
    
    note right of CompareContent
        SHA-256 hash comparison
        for change detection
    end note
    
    note right of UpdateEmbeddings
        GPU-accelerated processing
        Batch optimization
    end note
```

---

## 📋 Diagram Legend

### 🎨 Color Coding
- **Blue** (`#e3f2fd`): External systems and user interfaces
- **Purple** (`#f3e5f5`): Web layer components (frontend, API)
- **Orange** (`#fff3e0`): Processing and worker services
- **Green** (`#e8f5e8`): Storage and database systems
- **Pink** (`#fce4ec`): Infrastructure and DevOps components

### 🔗 Connection Types
- **Solid Lines**: Direct connections and data flow
- **Dashed Lines**: Indirect connections and dependencies
- **Arrows**: Direction of data flow or communication

### 📊 Component Types
- **Rectangles**: Services and applications
- **Circles**: Databases and storage
- **Diamonds**: Decision points and conditionals
- **Rounded Rectangles**: User interfaces and external systems

---

*This document provides comprehensive visual documentation of the KnowledgeHub architecture. For detailed implementation information, refer to the [Architecture Guide](ARCHITECTURE.md) and [API Documentation](API.md).*