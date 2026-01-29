# LLMs Engineering - Professional Training Program

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991.svg)](https://openai.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-000000.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In_Progress-yellow.svg)]()

## Overview

A comprehensive 8-project professional training program focused on Large Language Models (LLMs) engineering, from foundational concepts to advanced autonomous AI agents. This repository showcases hands-on projects demonstrating expertise in AI integration, prompt engineering, RAG systems, fine-tuning, and production-ready LLM applications.

**Based on:** [Ingeniería LLM - IA Generativa & Modelos de Lenguaje a Gran Escala](https://www.udemy.com/course/ingenieria-llm-ia-generativa-modelos-lenguaje-gran-escala-juan-gomila/) by Juan Gomila on Udemy

## Program Structure

The program is organized into **8 hands-on projects**, each demonstrating advanced LLM engineering skills and real-world business applications. The course covers frontier models (GPT-4, Claude) and open-source alternatives, comparing techniques like RAG, fine-tuning, and agentic workflows.

---

## Weekly Projects

### Project 1: Personalized Programming Tutor (Foundation)
**Status:** ✅ Completed  
**Skills Demonstrated:**
- OpenAI API integration
- Ollama local LLM deployment
- Streaming response implementation
- Interactive prompt engineering
- User input handling and validation

**Project Highlights:**
- Built an AI-powered programming tutor that explains complex code concepts
- Implemented real-time streaming with typewriter effect
- Supports multiple LLM providers (OpenAI GPT-4o-mini, Ollama)
- Interactive command-line interface for user queries
- Markdown-formatted educational responses

**Technologies:** `Python` `OpenAI API` `Ollama` `IPython` `Markdown`

[📂 View Project](./Project1-week1.ipynb)

---

### Project 2: FlightAI - Multimodal Airline Assistant
**Status:** ✅ Completed  
**Skills Demonstrated:**
- Function calling with tool execution.
- Multimodal AI (text, images, voice)
- Real-time audio transcription with Whisper
- Text-to-speech with speed control
- Context-aware image generation
- Interactive web UI with Gradio
- Voice input integration

**Project Highlights:**
- Built a professional multimodal customer service assistant for airlines
- Implemented 4 automated tools: ticket pricing, reservations, translation, audio transcription
- Integrated DeepSeek-V3.1 671B for natural language conversations with function calling
- Free image generation using Pollinations.AI for destination previews
- Voice interaction: microphone input → Whisper transcription → chat processing → TTS response
- Gradio interface with chat, image display, and audio playback
- Adjustable audio speed (1.1x) for faster responses

**Technologies:** `Python` `OpenAI Whisper` `gTTS` `Gradio` `Ollama DeepSeek` `Pollinations.AI` `pydub` `Function Calling`

[📂 View Project](./Project2-week2.ipynb)

---

### Project 3: Synthetic Data Studio - AI-Powered Dataset Generator
**Status:** ✅ Completed  
**Skills Demonstrated:**
- Model quantization (4-bit) with BitsAndBytes
- Local LLM deployment and optimization
- Structured prompt engineering for data generation
- Schema-driven AI outputs
- GPU memory management
- Pandas DataFrame manipulation
- Interactive web UI with Gradio
- Data validation and quality checks

**Project Highlights:**
- Built a production-ready synthetic data generator using Llama 3.1 8B (quantized to 4-bit)
- Implemented 3 business domain templates: Retail Sales, Bank Transactions, Customer Support
- Schema-driven generation with strict column specifications and data type constraints
- Automatic CSV parsing with robust error handling for LLM output quirks
- Quality validation: checks for missing columns, extra columns, and row count accuracy
- Memory-optimized inference with automatic GPU cache cleanup
- Gradio interface with schema selection, row count slider, and CSV download
- Running 8B parameter models on consumer GPUs (4GB VRAM vs 32GB required for FP32)

**Technologies:** `Python` `Hugging Face Transformers` `Llama 3.1 8B` `BitsAndBytes` `Gradio` `Pandas` `PyTorch` `4-bit Quantization`

[📂 View Project](./Project3-week3.ipynb)

---

### Project 4: Meeting Minutes & Action Items Generator
**Status:** 📋 Planned  
**Course Objective:** *Develop a tool that creates meeting minutes and action items from audio using open and closed models*

**Planned Skills:**
- Audio transcription (Whisper)
- Information extraction from transcripts
- Action item detection
- Summary generation
- Model comparison (open vs closed)

**Planned Technologies:** `Whisper` `GPT-4` `Llama` `Audio Processing` `NLP`

---

### Project 5: Python to C++ AI Optimizer
### Project 5: Private Knowledge Chatbot (RAG)
**Status:** ✅ Completed  
**Objective:** *Build a learning-focused RAG chatbot that answers questions using a private local knowledge base (PDFs) and a vector database.*

**Skills Demonstrated:**
- Document ingestion from a local knowledge base
- Chunking strategy (overlap + size trade-offs)
- Embeddings creation (OpenAI by default, HuggingFace optional)
- Vector database persistence with Chroma
- Conversational retrieval (LangChain) with memory
- Rapid prototyping with Gradio ChatInterface
- Embedding-space debugging via t-SNE (2D + 3D)

**Project Highlights:**
- Loads PDFs from `knowledge-base/`, splits them into chunks, and indexes them into `vector_db/`
- Retrieves top-$k$ relevant chunks per question and grounds the response on retrieved context
- Provides a simple interactive chat UI to validate behavior like an end user
- Includes visualization to build intuition and detect outliers in the vector store

**Technologies:** `Python` `LangChain` `ChromaDB` `OpenAI` `Gradio` `Plotly` `scikit-learn` `RAG`

[📂 View Project](./Project5-week5.ipynb)

---

### Project 6: AI Knowledge Worker with RAG
**Status:** 📋 Planned  
**Course Objective:** *Build an AI knowledge worker using RAG to become an expert in all company matters*

**Planned Skills:**
- Retrieval-Augmented Generation (RAG)
- Vector databases
- Semantic search
- Document indexing and retrieval
- Context-aware responses

**Planned Technologies:** `LangChain` `ChromaDB` `FAISS` `Sentence Transformers` `Pinecone`

---

### Project 7: Product Price Prediction (Part A & B)
**Status:** 📋 Planned  
**Course Objectives:**  
- *Part A: Predict product prices from short descriptions using Frontier models*
- *Part B: Fine-tune open-source models to compete with Frontier in price prediction*

**Planned Skills:**
- Prompt engineering for prediction tasks
- Model fine-tuning (LoRA, QLoRA)
- Performance comparison (Frontier vs Open-source)
- Dataset preparation
- Model evaluation and benchmarking

**Planned Technologies:** `GPT-4` `Claude` `Llama 3` `Hugging Face` `LoRA` `Weights & Biases`

---

### Project 8: Autonomous Multi-Agent Deal Detection System (Final Project)
**Status:** 📋 Planned  
**Course Objective:** *Build an autonomous multi-agent system that collaborates with models to detect deals and send notifications*

**Planned Skills:**
- Multi-agent architectures
- Agent collaboration and coordination
- Autonomous task execution
- Deal detection algorithms
- Notification systems
- Agentic workflows

**Planned Technologies:** `AutoGen` `LangGraph` `CrewAI` `Multiple LLMs` `Task Orchestration`

---

## Key Features

### 🎯 Professional Focus
- Production-ready code with best practices
- Comprehensive documentation
- Error handling and validation
- Performance optimization

### 🔧 Technical Stack
- **LLM Providers:** OpenAI (GPT-4, GPT-4o-mini), Ollama (Local models)
- **Frameworks:** LangChain, AutoGen, LangGraph
- **Languages:** Python 3.10+
- **Tools:** Jupyter Notebooks, VS Code, Docker

### 📊 Skills Development
- API integration and management
- Streaming and async operations
- Prompt engineering and optimization
- Vector databases and embeddings
- Agent architectures and orchestration
- Production deployment strategies

---

## Getting Started

### Prerequisites

```bash
# Python 3.10 or higher
python --version

# Virtual environment (recommended)
python -m venv llms-env
source llms-env/bin/activate  # On Windows: llms-env\Scripts\activate
```

### How to run (notebook-first workflow)

This repository is primarily **Jupyter notebooks**. The recommended workflow is:

1) Open the project notebook (e.g., `Project5-week5.ipynb`) in VS Code or Jupyter.
2) Ensure your Python environment has the required packages (install them *from the notebook* if needed).
3) Run the cells top-to-bottom.

> Note: There is no `requirements.txt` at the repository root. Dependencies are typically installed from within notebooks as you work through the program.

### Configuration

Create a `.env` file (recommended location for container-based notebooks: `/workspace/.env`).

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=llama3.2

# Ollama Cloud (for DeepSeek-V3.1)
OLLAMA_BASE_URL=https://cloud.ollamaapi.com
OLLAMA_API_KEY=your_ollama_cloud_key

# Provider Selection
LLM_PROVIDER=ollama  # or 'openai'
```

---

## Project Structure

```
llms-engineering-main/
├── Project1-week1.ipynb          # Week 1: Personalized Programming Tutor
├── Project2-week2.ipynb          # Week 2: FlightAI - Multimodal Airline Assistant
├── Project3-week3.ipynb          # Week 3: Synthetic Data Studio
├── Project5-week5.ipynb          # Week 5: Private Knowledge Chatbot (RAG)
├── knowledge-base/               # Local documents used for RAG (PDFs)
├── vector_db/                    # Local Chroma persistence (generated)
├── week5/                        # Week 5 lessons + additional KB/DB assets
└── README.md                     # This file
```

---

## Week 5 — What I learned

- How to build a complete RAG loop: **ingest → chunk → embed → store → retrieve → generate**
- Why chunking matters (size/overlap trade-offs) and how it affects retrieval quality and cost
- How Chroma persists embeddings locally and how to keep experiments repeatable
- How to prototype a usable chatbot UI quickly with Gradio
- How embedding visualizations (t-SNE) can help debug indexing issues (outliers, empty chunks, odd clusters)

---

## Usage Examples

### Week 1 - Programming Tutor

```python
# Execute the notebook cells in order
# Cell 7 prompts for your code question:

# Example input:
yield from {book.get("author") for book in books if book.get("author")}

# The tutor provides a detailed explanation with:
# - What the code does
# - Step-by-step breakdown
# - Why use this approach
# - Important concepts involved
```

### Week 2 - FlightAI Assistant

```python
# Launch the Gradio interface by executing all cells
# Three interaction modes:

# 1. Text Input:
"What's the price to Paris?"
# → Assistant calls get_ticket_price tool
# → Generates image of Paris
# → Responds with audio

# 2. Voice Input:
# Click microphone → speak your question → auto-transcription
# → Same processing as text input

# 3. Example queries:
"Make a reservation for John Doe to Tokyo on 2025-12-01"
"Translate 'Hello' to Spanish"
```

### Week 3 - Synthetic Data Studio

```python
# Launch the Gradio interface by executing all cells
# Configure your synthetic dataset:

# 1. Select Dataset Type (dropdown):
"Retail Sales"  # or "Bank Transactions", "Customer Support Tickets"

# 2. Set Row Count (slider):
100  # Between 10 and 1000 rows

# 3. Add Custom Instructions (optional):
"Generate 10% fraudulent transactions"
"Ensure balanced distribution across countries"

# Click "Generate Synthetic Data 🚀"
# → LLM constructs dataset according to schema
# → Output parsed to pandas DataFrame
# → Quality validation performed
# → CSV file ready for download

# Example generated columns (Retail Sales):
# order_id, order_date, customer_id, country, product_category,
# unit_price, quantity, total_amount, is_fraud
```

---

## Technical Highlights

### Streaming Implementation
- Real-time response rendering with typewriter effect
- Efficient memory management for long responses
- Graceful error handling

### Multi-Provider Support
- Seamless switching between OpenAI and Ollama
- Unified interface for different LLM backends
- Cost optimization with local models

### Model Quantization & Optimization
- 4-bit quantization using BitsAndBytes (8GB → 4GB VRAM)
- Local deployment of billion-parameter models on consumer hardware
- GPU memory management and cache optimization
- Automatic device mapping for multi-GPU setups

### Production Considerations
- Environment-based configuration
- API key security best practices
- Error logging and monitoring
- Rate limiting awareness

---

## Learning Outcomes

By completing this 8-project program, you will demonstrate:

### Core Skills
✅ **Design & Develop Complete Solutions** - Build end-to-end LLM solutions for business problems  
✅ **Model Selection & Evaluation** - Compare top 10 Frontier and open-source LLMs, selecting the best for each task  
✅ **Advanced Techniques** - Apply RAG, fine-tuning, and agentic workflows to improve performance  
✅ **Platform Mastery** - Work with Hugging Face, Gradio, and Weights & Biases  

### Technical Expertise
✅ **API Integration** - Master both Frontier (GPT-4, Claude) and open-source LLMs via API and direct inference  
✅ **Prompt Engineering** - Craft effective system and user prompts for optimal results  
✅ **Streaming & Async Operations** - Handle real-time response rendering  
✅ **RAG Systems** - Build context-aware AI applications with vector databases  
✅ **Fine-Tuning** - Train and optimize models for specific tasks  
✅ **Multi-Agent Systems** - Create autonomous collaborative AI agents  
✅ **Production Deployment** - Take projects from development to production  

### Fundamental Concepts
✅ **AI Paradigms** - Define common AI paradigms and match them to business problems  
✅ **Deep Learning Foundations** - Understand training vs inference, generalization, and optimization  
✅ **Generative AI Concepts** - Explain LLMs, Transformer Architecture, and their capabilities  
✅ **LLM Internals** - Understand how LLMs work in sufficient detail to train, test, and troubleshoot  

### Practical Applications
✅ **Code Generation** - Write documents, answer questions, and generate images  
✅ **Performance Optimization** - Achieve 60,000x performance improvements through code optimization  
✅ **Multimodal AI** - Work with text, images, and audio  
✅ **Business Value** - Solve real-world business problems with AI solutions

---

## Technologies & Tools

| Category | Tools |
|----------|-------|
| **Frontier LLMs** | OpenAI (GPT-4, GPT-4o), Anthropic (Claude 3.5) |
| **Open-Source LLMs** | Llama 3, Code Llama, Mistral, Gemma |
| **Frameworks** | LangChain, AutoGen, LangGraph, CrewAI |
| **Vector DBs** | ChromaDB, FAISS, Pinecone |
| **Fine-Tuning** | Hugging Face, LoRA, QLoRA, Weights & Biases |
| **Languages** | Python 3.10+ |
| **Development** | Jupyter, VS Code, Git, Gradio |
| **Deployment** | Docker, FastAPI, AWS/GCP |
| **Audio/Vision** | Whisper, GPT-4V, DALL-E |

---

## Course Credits

This repository is based on the comprehensive Udemy course:

**[Ingeniería LLM - IA Generativa & Modelos de Lenguaje a Gran Escala](https://www.udemy.com/course/ingenieria-llm-ia-generativa-modelos-lenguaje-gran-escala-juan-gomila/)**  
**Instructor:** Juan Gomila  
**Platform:** Udemy

### Course Highlights:
- 8 hands-on projects solving real business problems
- Comparison of Frontier models (GPT-4, Claude) vs open-source alternatives
- Advanced techniques: RAG, fine-tuning, agentic workflows
- From fundamentals to autonomous multi-agent systems
- Production-ready implementations with best practices

All projects in this repository are implementations and adaptations of the course curriculum, demonstrating practical application of learned concepts with personal enhancements and professional code standards.

---

## Contributing

This is a personal learning repository showcasing professional LLM engineering skills developed through the course. Suggestions and feedback are welcome!

---

## Acknowledgments

- **Juan Gomila** - Course instructor and content creator ([Udemy Course](https://www.udemy.com/course/ingenieria-llm-ia-generativa-modelos-lenguaje-gran-escala-juan-gomila/))
- **OpenAI** - GPT-4, GPT-4o, and comprehensive API documentation
- **Anthropic** - Claude 3.5 and advanced AI safety research
- **Ollama** - Local LLM deployment and optimization tools
- **Hugging Face** - Open-source model hub and fine-tuning ecosystem
- **The LLM Engineering Community** - Best practices, patterns, and continuous learning

---
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated:** December 2025  
**Last Updated:** January 2026  
**Program Status:** Projects 1-3 & 5 Completed | 4 Projects Remaining

