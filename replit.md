# EHR Query Agent - Supabase Patient-Locked

## Overview
This is a Streamlit-based Electronic Health Record (EHR) Query Agent that uses Retrieval-Augmented Generation (RAG) to provide AI-powered responses about patient medical records. The application connects to Supabase for data storage and uses OpenAI for embeddings and chat completion.

## Project Purpose
The application allows healthcare professionals to:
- Search and lock specific patient records securely
- Ask natural language questions about patient medical information
- Get AI-powered responses based on retrieved patient data
- View sources for all AI responses for transparency

## Recent Changes
- **2025-10-23**: Initial setup in Replit environment
  - Configured Python 3.11 environment
  - Installed all dependencies from requirements.txt
  - Set up Streamlit server on port 5000
  - Configured deployment settings
  - Added .gitignore and .env.example files

## Project Architecture

### Tech Stack
- **Frontend**: Streamlit (Python web framework)
- **Database**: Supabase (PostgreSQL with vector search)
- **AI/ML**: 
  - OpenAI GPT-4o-mini for chat completion
  - OpenAI text-embedding-3-small for semantic search
  - LangChain for AI orchestration
- **Search**: Vector similarity search using Supabase pgvector

### File Structure
```
.
├── app.py                   # Main Streamlit application
├── patients.py              # Patient roster and fuzzy matching logic
├── retrieve_supabase.py     # RAG retrieval functions
├── supabase_client.py       # Supabase client initialization
├── ingest_sample.py         # Sample data ingestion script
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── .env.example            # Environment variables template
└── .gitignore              # Git ignore rules
```

### Key Components

1. **app.py**: Main application with patient selection and chat interface
2. **patients.py**: Handles patient roster building and fuzzy name matching
3. **retrieve_supabase.py**: Implements vector similarity search for RAG
4. **supabase_client.py**: Supabase connection management
5. **ingest_sample.py**: Utility script to ingest sample patient data

## Configuration

### Required Environment Variables
The following secrets are configured in Replit Secrets:
- `OPENAI_API_KEY`: OpenAI API key for embeddings and chat
- `SUPABASE_URL`: Your Supabase project URL
- `SERVICE_SUPABASEANON_KEY`: Supabase anonymous key (for reads)
- `SERVICE_SUPABASESERVICE_KEY`: Supabase service role key (for writes)

### Optional Environment Variables
- `LLM_MODEL`: OpenAI model to use (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)

## Running the Application

### Development
The Streamlit app runs automatically via the configured workflow on port 5000.

### Deployment
The application is configured for autoscale deployment. Click the "Deploy" button in Replit to publish your app.

## Database Schema

The application expects a Supabase table `rag_chunks` with the following structure:
- `id`: Primary key
- `content`: Text content of the medical record chunk
- `metadata`: JSONB containing patient information
- `embedding`: Vector(1536) for semantic search

Required SQL function: `match_patient_chunks_arr` for vector similarity search.

## Usage Flow

1. **Load Patient Roster**: App queries Supabase to build list of patients
2. **Patient Selection**: User searches by ID or name (fuzzy matching supported)
3. **DOB Verification**: User confirms patient identity with date of birth
4. **Patient Lock**: System locks to selected patient for secure querying
5. **Chat Interface**: User asks questions about the locked patient
6. **RAG Retrieval**: System retrieves relevant chunks from Supabase
7. **AI Response**: OpenAI generates response based on retrieved context
8. **Source Display**: Sources are shown for transparency

## Security Features

- Patient verification via DOB before access
- Patient-locked queries (can only query one patient at a time)
- Environment variables for sensitive credentials
- No exposure of API keys in code

## Development Notes

- The app uses Streamlit's session state to manage patient locks
- Vector embeddings are 1536-dimensional (OpenAI text-embedding-3-small)
- CORS and XSRF protection disabled for Replit proxy compatibility
- Server runs on 0.0.0.0:5000 for proper Replit routing
