# EHR Query Agent

A Streamlit-based medical EHR query system that uses LLMs to answer questions about patients and general medical knowledge.

## Features

- 🏥 Automatic patient vs. general query routing using SLM
- 👥 Multi-patient comparison support
- 📊 Graph/chart generation capabilities
- 🔍 Semantic search using Supabase vector embeddings
- 💬 Conversational memory

## Quick Start

### Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SERVICE_SUPABASEANON_KEY=your_supabase_key
   LLM_MODEL=gpt-4o-mini  # Optional
   ROUTER_MODEL=gpt-3.5-turbo  # Optional, for routing
   EMBEDDING_MODEL=text-embedding-3-small  # Optional
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Add secrets in "Secrets" tab:
     ```
     OPENAI_API_KEY=your_key
     SUPABASE_URL=your_url
     SERVICE_SUPABASEANON_KEY=your_key
     LLM_MODEL=gpt-4o-mini
     ROUTER_MODEL=gpt-3.5-turbo
     EMBEDDING_MODEL=text-embedding-3-small
     ```
   - Click "Deploy"

### Option 2: Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t ehr-query-agent .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 \
     -e OPENAI_API_KEY=your_key \
     -e SUPABASE_URL=your_url \
     -e SERVICE_SUPABASEANON_KEY=your_key \
     ehr-query-agent
   ```

3. **Or use docker-compose:**
   Create `docker-compose.yml`:
   ```yaml
   version: '3.8'
   services:
     app:
       build: .
       ports:
         - "8501:8501"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - SUPABASE_URL=${SUPABASE_URL}
         - SERVICE_SUPABASEANON_KEY=${SERVICE_SUPABASEANON_KEY}
   ```

### Option 3: Other Platforms

- **Railway**: Connect GitHub repo, set environment variables
- **Render**: Connect GitHub repo, use Dockerfile, set environment variables
- **AWS/GCP/Azure**: Use Dockerfile with their container services

## Project Structure

- `app.py` - Main Streamlit application
- `query_analyzer.py` - SLM-based query routing
- `patients.py` - Patient roster management
- `retrieve_supabase.py` - Vector search functions
- `supabase_client.py` - Supabase connection

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SERVICE_SUPABASEANON_KEY` | Yes | Supabase anon/service key |
| `LLM_MODEL` | No | Main LLM model (default: gpt-4o-mini) |
| `ROUTER_MODEL` | No | Routing SLM (default: gpt-3.5-turbo) |
| `EMBEDDING_MODEL` | No | Embedding model (default: text-embedding-3-small) |

## Notes

- Ensure your Supabase database has the `rag_chunks` table with vector embeddings
- The app requires patient data in `rag_chunks.metadata` with `patient_id`, `first_name`, `last_name`, `dob` fields

