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

[📂 View Project](./week1-Project1.ipynb)

---

### Project 2: FlightAI - Multimodal Airline Assistant
**Status:** ✅ Completed  
**Skills Demonstrated:**
- Function calling with tool execution
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

[📂 View Project](./week2-Project2.ipynb)

---

### Project 3: AI-Powered Brochure Generator
**Status:** 📋 Planned  
**Course Objective:** *Build an AI brochure generator that intelligently scrapes and navigates company websites*

**Planned Skills:**
- Web scraping and navigation
- Intelligent content extraction
- Multi-page data aggregation
- Generative AI for content creation
- Automated brochure formatting

**Planned Technologies:** `BeautifulSoup` `Selenium` `OpenAI GPT-4` `Web Scraping APIs`

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
**Status:** 📋 Planned  
**Course Objective:** *Create an AI that converts Python code to optimized C++, increasing performance by 60,000x!*

**Planned Skills:**
- Code translation
- Performance optimization
- Language model fine-tuning for code
- Benchmarking and testing
- Compilation optimization

**Planned Technologies:** `GPT-4` `Code Llama` `C++ Compilers` `Performance Profiling`

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

### Installation

```bash
# Clone repository
git clone https://github.com/oscargbocanegra/llms-engineering-main.git
cd llms-engineering-main

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file in the workspace root:

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
├── week1-Project1.ipynb          # Week 1: Personalized Programming Tutor
├── week2-Project2.ipynb          # Week 2: FlightAI - Multimodal Airline Assistant
├── week3-Project3.ipynb          # Week 3: AI Brochure Generator (Planned)
├── week4-Project4.ipynb          # Week 4: Meeting Minutes Generator (Planned)
├── week5-Project5.ipynb          # Week 5: Python to C++ Optimizer (Planned)
├── week6-Project6.ipynb          # Week 6: RAG Knowledge Worker (Planned)
├── week7-Project7.ipynb          # Week 7: Price Prediction (Planned)
├── week8-Project8.ipynb          # Week 8: Multi-Agent Deal Detection (Planned)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

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

**Last Updated:** November 2025  
**Program Status:** Projects 1-2 Completed | 6 Projects Remaining

