# RAGDoc - Intelligent Document Chat Application

A sophisticated Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and have intelligent conversations with their content using either Mistral AI or Google Gemini models.

## 🚀 Features

- **Multi-Provider AI Support**: Choose between Mistral AI and Google Gemini for chat completions
- **PDF Document Upload**: Easy drag-and-drop PDF upload with automatic text extraction
- **Intelligent Search**: Vector-based document search using FAISS indexing
- **Hybrid Information**: Combines document content with web search results for comprehensive answers
- **Real-time Chat**: Interactive chat interface with document references
- **Fallback Mechanisms**: Automatic fallbacks for API availability and error handling
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   External APIs │
│   (Next.js)     │    │   (FastAPI)     │    │                 │
│                 │    │                 │    │                 │
│ • Chat UI       │◄──►│ • Document Mgmt │◄──►│ • Mistral AI    │
│ • File Upload   │    │ • Vector Search │    │ • Google Gemini │
│ • Provider      │    │ • RAG Pipeline  │    │ • DuckDuckGo    │
│   Selection     │    │ • Web Search    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.10+** (backend)
- **Node.js 18+** (frontend)
- **API Keys**:
  - Mistral AI API key (optional)
  - Google Gemini API key (required)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RAGDoc
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 3. Frontend Setup
```bash
cd my-app

# Install dependencies
npm install
```

## ⚙️ Configuration

Create a `.env` file in the `backend` directory:

```env
# Mistral AI Configuration (optional)
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_CHAT_MODEL=mistral-large-latest
MISTRAL_EMBED_MODEL=mistral-embed

# Google Gemini Configuration (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Embedding Configuration
EMBED_BATCH_MAX_TOKENS=6000
EMBED_BATCH_MAX_ITEMS=32
EMBED_ITEM_MAX_TOKENS=1200
```

## 🚀 Running the Application

### Start Backend Server
```bash
cd backend
source .venv/bin/activate
export $(grep -v '^#' .env | xargs) && python main.py
```
Backend will run on `http://localhost:8000`

### Start Frontend Server
```bash
cd my-app
npm run dev
```
Frontend will run on `http://localhost:3000`

## 📖 Usage

1. **Upload Documents**: Drag and drop PDF files onto the upload area
2. **Select Provider**: Choose between Mistral AI or Google Gemini in the chat interface
3. **Start Chatting**: Ask questions about your uploaded documents
4. **Get Answers**: Receive AI-generated responses with document references and citations

## 🔧 API Endpoints

### Backend Endpoints

- `GET /health` - Health check
- `GET /documents` - List uploaded documents
- `POST /upload` - Upload PDF document
- `POST /chat` - Chat with documents
- `GET /files/{file_id}` - Serve uploaded files

### Chat Request Format
```json
{
  "docId": "document-uuid",
  "message": "Your question here",
  "max_chunks": 4,
  "provider": "mistral" | "gemini"
}
```

## 🏛️ Technical Stack

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **React Context**: State management

### Backend
- **FastAPI**: Modern Python web framework
- **FAISS**: Vector similarity search
- **PyPDF**: PDF text extraction
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### AI Services
- **Mistral AI**: Chat completions and embeddings
- **Google Gemini**: Chat completions and embeddings
- **DuckDuckGo**: Web search integration

## 🔄 Workflow

1. **Document Upload**: PDF files are uploaded and processed
2. **Text Extraction**: PyPDF extracts text from each page
3. **Chunking**: Text is split into manageable chunks
4. **Embedding**: Text chunks are converted to vectors using AI embeddings
5. **Indexing**: FAISS creates searchable vector indexes
6. **Query Processing**: User queries are embedded and matched against document vectors
7. **Context Retrieval**: Relevant document chunks are retrieved
8. **Web Enhancement**: Optional web search provides additional context
9. **AI Generation**: Selected AI provider generates responses using retrieved context

## 🛡️ Error Handling

- **API Fallbacks**: Automatic fallback between Mistral and Gemini
- **Graceful Degradation**: Continues operation with excerpts if AI APIs fail
- **Validation**: Input validation and sanitization
- **Error Messages**: User-friendly error reporting

## 📁 Project Structure

```
RAGDoc/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── .env                 # Environment variables
│   └── storage/             # Document storage
│       ├── files/           # Uploaded PDFs
│       ├── indexes/         # FAISS indexes
│       └── meta/            # Document metadata
├── my-app/
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   └── contexts/        # React contexts
│   ├── package.json         # Node.js dependencies
│   └── tailwind.config.js   # Tailwind configuration
└── README.md                # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **FAISS Installation**: If FAISS fails to install, use conda:
   ```bash
   conda install faiss-cpu=1.8.0
   ```

2. **API Key Issues**: Ensure your API keys are correctly set in the `.env` file

3. **Port Conflicts**: If ports 3000 or 8000 are in use, modify the startup commands

4. **Dependencies**: Ensure Python 3.10+ and Node.js 18+ are installed

### Getting Help

- Check the console logs for detailed error messages
- Verify API key validity and quotas
- Ensure all dependencies are properly installed

## 🔮 Future Enhancements

- Support for additional document formats (DOCX, TXT, etc.)
- Multi-language support
- Advanced filtering and search capabilities
- User authentication and document management
- Cloud deployment configurations