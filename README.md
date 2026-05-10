# 📄 PDF RAG Chatbot

A Streamlit-based chatbot application that enables you to upload PDF documents and ask questions about their contents using Retrieval-Augmented Generation (RAG) powered by Google's Gemini API.

## Features

- **PDF Upload**: Easily upload PDF files to extract and index their content
- **RAG Architecture**: Uses FAISS vector database for efficient semantic search and retrieval
- **AI-Powered Answers**: Leverages Google's Gemini 2.5 Flash model for intelligent question answering
- **Conversation History**: Maintains chat history for context-aware responses
- **Real-time Processing**: Processes and indexes PDFs on-the-fly

## Technology Stack

- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration and RAG pipeline
- **FAISS**: Vector database for semantic search
- **Google Gemini API**: Large language model and embeddings
- **PyPDF**: PDF document loading
- **Python Dotenv**: Environment variable management

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([get one here](https://aistudio.google.com/app/apikeys))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eboekenh/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Google API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

Alternatively, you can enter your API key directly in the application when prompted.

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload a PDF file using the file uploader

4. Enter your Google Gemini API key (if not already set in `.env`)

5. Ask questions about the PDF content in the chat interface

## How It Works

1. **PDF Processing**: The uploaded PDF is split into chunks (1000 tokens with 200 token overlap)
2. **Embeddings**: Each chunk is converted to embeddings using Google's Gemini embedding model
3. **Vector Indexing**: Embeddings are stored in a FAISS index for fast retrieval
4. **Query Answering**: User questions are converted to embeddings and matched against the PDF chunks
5. **Context Generation**: Retrieved relevant chunks are passed to the Gemini model for answer generation

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore rules
└── faiss/                # Vector index storage (auto-created)
```

## Configuration

You can modify the following parameters in `app.py`:

- **Chunk Size**: Default 1000 tokens (line 28)
- **Chunk Overlap**: Default 200 tokens (line 28)
- **LLM Model**: Default `gemini-2.5-flash` (line 42)
- **Temperature**: Default 0.3 for more deterministic answers (line 42)
- **Retrieval K**: Default 5 most relevant chunks (line 43)

## Troubleshooting

### "Permission denied" or API errors
- Verify your Google Gemini API key is valid and has the required permissions
- Check that the API is enabled in your Google Cloud project

### Vector index not found
- The FAISS index is created dynamically when a PDF is uploaded
- If issues persist, delete the `faiss/` directory and re-upload the PDF

### Out of memory errors
- Large PDFs may require more RAM
- Consider breaking PDFs into smaller files or increasing chunk overlap

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
