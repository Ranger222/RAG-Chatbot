# RAGDoc Architecture Documentation

## 🏗️ System Overview

RAGDoc is a full-stack Retrieval-Augmented Generation (RAG) application that enables intelligent conversations with PDF documents using state-of-the-art AI models.

## 📊 High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Next.js UI]
        CTX[React Context]
        COMP[React Components]
    end
    
    subgraph "Backend Layer"
        API[FastAPI Server]
        DOC[Document Manager]
        VEC[Vector Search]
        RAG[RAG Pipeline]
    end
    
    subgraph "Storage Layer"
        FILES[File Storage]
        META[Metadata Store]
        FAISS[FAISS Index]
    end
    
    subgraph "External Services"
        MISTRAL[Mistral AI]
        GEMINI[Google Gemini]
        WEB[DuckDuckGo Search]
    end
    
    UI --> API
    CTX --> UI
    COMP --> CTX
    
    API --> DOC
    API --> VEC
    API --> RAG
    
    DOC --> FILES
    DOC --> META
    VEC --> FAISS
    
    RAG --> MISTRAL
    RAG --> GEMINI
    RAG --> WEB
```

## 🔄 Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant AI as AI Service
    participant S as Storage
    
    Note over U,S: Document Upload Flow
    U->>F: Upload PDF
    F->>B: POST /upload
    B->>B: Extract text
    B->>B: Create chunks
    B->>AI: Generate embeddings
    AI-->>B: Vector embeddings
    B->>S: Store document + index
    B-->>F: Upload success
    F-->>U: Confirmation
    
    Note over U,S: Chat Flow
    U->>F: Send message
    F->>B: POST /chat
    B->>AI: Embed query
    AI-->>B: Query vector
    B->>S: Search FAISS index
    S-->>B: Relevant chunks
    B->>AI: Generate response
    AI-->>B: AI response
    B-->>F: Chat response
    F-->>U: Display answer
```

## 🏛️ Component Architecture

### Frontend Architecture

```mermaid
graph TD
    subgraph "Next.js Application"
        APP[App Router]
        LAYOUT[Root Layout]
        
        subgraph "Pages"
            HOME[Home Page]
            CHAT[Chat Page]
        end
        
        subgraph "Components"
            UPLOAD[Upload Component]
            CHATUI[Chat UI]
            DOCLIST[Document List]
            PROVIDER[Provider Selector]
        end
        
        subgraph "Context"
            DOCSCTX[Docs Context]
            STATE[Application State]
        end
    end
    
    APP --> LAYOUT
    LAYOUT --> HOME
    LAYOUT --> CHAT
    
    HOME --> UPLOAD
    HOME --> DOCLIST
    CHAT --> CHATUI
    CHAT --> PROVIDER
    
    UPLOAD --> DOCSCTX
    DOCLIST --> DOCSCTX
    CHATUI --> DOCSCTX
    PROVIDER --> DOCSCTX
    
    DOCSCTX --> STATE
```

### Backend Architecture

```mermaid
graph TD
    subgraph "FastAPI Application"
        APP[FastAPI App]
        CORS[CORS Middleware]
        STATIC[Static Files]
        
        subgraph "API Endpoints"
            HEALTH[/health]
            DOCS[/documents]
            UPLOAD[/upload]
            CHAT[/chat]
            FILES[/files]
        end
        
        subgraph "Core Services"
            DOCMGR[Document Manager]
            EMBED[Embedding Service]
            SEARCH[Search Service]
            RAGPIPE[RAG Pipeline]
        end
        
        subgraph "AI Integrations"
            MISTRALAI[Mistral Client]
            GEMINIAI[Gemini Client]
            WEBSEARCH[Web Search]
        end
        
        subgraph "Storage"
            FILESYSTEM[File System]
            FAISSDB[FAISS Database]
            METADATA[JSON Metadata]
        end
    end
    
    APP --> CORS
    APP --> STATIC
    APP --> HEALTH
    APP --> DOCS
    APP --> UPLOAD
    APP --> CHAT
    APP --> FILES
    
    UPLOAD --> DOCMGR
    CHAT --> RAGPIPE
    DOCS --> DOCMGR
    
    DOCMGR --> EMBED
    RAGPIPE --> SEARCH
    RAGPIPE --> MISTRALAI
    RAGPIPE --> GEMINIAI
    RAGPIPE --> WEBSEARCH
    
    EMBED --> MISTRALAI
    EMBED --> GEMINIAI
    SEARCH --> FAISSDB
    DOCMGR --> FILESYSTEM
    DOCMGR --> METADATA
```

## 🛠️ Technical Stack Details

### Frontend Stack
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS for utility-first styling
- **State Management**: React Context API
- **HTTP Client**: Fetch API with async/await

### Backend Stack
- **Framework**: FastAPI for high-performance API
- **Language**: Python 3.10+ with type hints
- **PDF Processing**: PyPDF for text extraction
- **Vector Database**: FAISS for similarity search
- **Validation**: Pydantic for data models
- **Server**: Uvicorn ASGI server

### AI/ML Stack
- **Embedding Models**:
  - Mistral Embed (mistral-embed)
  - Google Text Embedding 004
- **Chat Models**:
  - Mistral Large Latest
  - Google Gemini 1.5 Flash
- **Vector Operations**: NumPy for numerical computing

## 📋 RAG Pipeline Workflow

```mermaid
flowchart TD
    START([User Query]) --> EMBED[Embed Query]
    EMBED --> SEARCH[Vector Search]
    SEARCH --> RETRIEVE[Retrieve Chunks]
    RETRIEVE --> WEB{Web Search?}
    WEB -->|Yes| WEBSEARCH[DuckDuckGo Search]
    WEB -->|No| CONTEXT[Build Context]
    WEBSEARCH --> CONTEXT
    CONTEXT --> PROVIDER{Select Provider}
    PROVIDER -->|Mistral| MISTRAL[Mistral API]
    PROVIDER -->|Gemini| GEMINI[Gemini API]
    MISTRAL --> RESPONSE[Generate Response]
    GEMINI --> RESPONSE
    RESPONSE --> FORMAT[Format Output]
    FORMAT --> END([Return to User])
    
    style START fill:#e1f5fe
    style END fill:#e8f5e8
    style RESPONSE fill:#fff3e0
```

## 🔧 Document Processing Pipeline

```mermaid
flowchart LR
    UPLOAD[PDF Upload] --> EXTRACT[Text Extraction]
    EXTRACT --> CHUNK[Text Chunking]
    CHUNK --> EMBED[Generate Embeddings]
    EMBED --> INDEX[FAISS Indexing]
    INDEX --> STORE[Store Metadata]
    STORE --> READY[Ready for Search]
    
    subgraph "Processing Details"
        EXTRACT --> |PyPDF| PAGES[Page-by-Page]
        CHUNK --> |Split by Tokens| SEGMENTS[Text Segments]
        EMBED --> |AI Service| VECTORS[Vector Arrays]
        INDEX --> |FAISS| SEARCHABLE[Searchable Index]
    end
```

## 🗄️ Data Models

### Core Data Structures

```python
# Document Model
class Document(BaseModel):
    id: str                    # UUID
    name: str                  # Original filename
    uploadedOn: str           # ISO timestamp
    size: int                 # File size in bytes
    type: str = "pdf"         # Document type
    fileUrl: Optional[str]    # Serving URL

# Chat Request
class ChatRequest(BaseModel):
    docId: str                # Document UUID
    message: str              # User query
    max_chunks: int = 4       # Max retrieved chunks
    provider: str = "mistral" # AI provider

# Chat Response
class ChatResponse(BaseModel):
    answer: str               # AI generated answer
    reference: Optional[ChatReference] # Doc reference
    origin: str = "document"  # Source type
    sources: Optional[List[WebSource]] # Web sources
```

### Storage Structure

```
storage/
├── documents.json          # Document catalog
├── files/                  # Original PDF files
│   └── {doc_id}.pdf
├── indexes/                # FAISS vector indexes
│   └── {doc_id}.faiss
└── meta/                   # Document metadata
    └── {doc_id}.json
```

## 🔀 Provider Fallback Strategy

```mermaid
flowchart TD
    CHAT[Chat Request] --> CHECK_PROVIDER{Provider Available?}
    CHECK_PROVIDER -->|Yes| PRIMARY[Use Selected Provider]
    CHECK_PROVIDER -->|No| FALLBACK[Try Alternative Provider]
    FALLBACK --> CHECK_ALT{Alternative Available?}
    CHECK_ALT -->|Yes| SECONDARY[Use Alternative]
    CHECK_ALT -->|No| EXCERPTS[Return Raw Excerpts]
    
    PRIMARY --> SUCCESS[Generate Response]
    SECONDARY --> SUCCESS
    EXCERPTS --> DEGRADED[Degraded Response]
    
    SUCCESS --> RETURN[Return to User]
    DEGRADED --> RETURN
```

## 🛡️ Error Handling Strategy

### API Error Handling

```mermaid
flowchart TD
    REQUEST[API Request] --> VALIDATE[Validate Input]
    VALIDATE --> PROCESS[Process Request]
    PROCESS --> ERROR{Error Occurred?}
    ERROR -->|No| SUCCESS[Return Success]
    ERROR -->|Yes| TYPE{Error Type?}
    
    TYPE -->|Validation| VALIDATION_ERROR[400 Bad Request]
    TYPE -->|Not Found| NOT_FOUND[404 Not Found]
    TYPE -->|External API| API_ERROR[502 Bad Gateway]
    TYPE -->|Server| SERVER_ERROR[500 Internal Error]
    
    VALIDATION_ERROR --> LOG[Log Error]
    NOT_FOUND --> LOG
    API_ERROR --> LOG
    SERVER_ERROR --> LOG
    
    LOG --> RESPONSE[Error Response]
    SUCCESS --> RESPONSE
```

### Embedding Service Fallback

```mermaid
flowchart TD
    EMBED_REQUEST[Embedding Request] --> MISTRAL_CHECK{Mistral Available?}
    MISTRAL_CHECK -->|Yes| MISTRAL[Use Mistral Embedding]
    MISTRAL_CHECK -->|No| GEMINI_CHECK{Gemini Available?}
    GEMINI_CHECK -->|Yes| GEMINI[Use Gemini Embedding]
    GEMINI_CHECK -->|No| ERROR[No Embedding Service]
    
    MISTRAL --> EMBED_SUCCESS[Embeddings Generated]
    GEMINI --> EMBED_SUCCESS
    ERROR --> FAIL[Return Error]
```

## 🚀 Performance Optimizations

### Vector Search Optimization
- **Batch Processing**: Process multiple embeddings in batches
- **Token Limits**: Respect API token limits for efficiency
- **Caching**: Cache computed embeddings and indexes
- **Chunking Strategy**: Optimal text chunk sizes for retrieval

### Frontend Optimization
- **Code Splitting**: Lazy load components
- **State Management**: Efficient React Context usage
- **API Caching**: Cache document lists and metadata
- **Responsive Design**: Optimized for all device sizes

### Backend Optimization
- **Async Operations**: Non-blocking I/O operations
- **Connection Pooling**: Efficient HTTP client usage
- **Memory Management**: Proper cleanup of large objects
- **Static File Serving**: Direct file serving with FastAPI

## 🔒 Security Considerations

### API Security
- **CORS Configuration**: Proper cross-origin settings
- **Input Validation**: Pydantic model validation
- **File Upload Limits**: Size and type restrictions
- **Error Sanitization**: No sensitive data in error messages

### Data Security
- **API Key Management**: Environment variable storage
- **File Isolation**: Sandboxed file storage
- **Access Control**: No direct file system access
- **Data Validation**: Comprehensive input sanitization

## 📈 Scalability Considerations

### Horizontal Scaling
- **Stateless Design**: No server-side sessions
- **External Storage**: File system can be replaced with cloud storage
- **Load Balancing**: Multiple backend instances supported
- **Database Migration**: FAISS can be replaced with vector databases

### Vertical Scaling
- **Memory Optimization**: Efficient vector storage
- **CPU Utilization**: Optimized embedding computations
- **Disk I/O**: Efficient file operations
- **Network Optimization**: Minimized API calls

## 🔮 Future Architecture Enhancements

### Planned Improvements
1. **Microservices**: Split into specialized services
2. **Vector Database**: Migrate from FAISS to Pinecone/Weaviate
3. **Caching Layer**: Redis for improved performance
4. **Authentication**: User management and access control
5. **Monitoring**: Comprehensive logging and metrics
6. **CI/CD Pipeline**: Automated testing and deployment

### Technology Evolution
- **Streaming Responses**: Real-time response generation
- **Multi-modal Support**: Images and tables in documents
- **Advanced RAG**: Hybrid search and re-ranking
- **Cloud Native**: Kubernetes deployment support