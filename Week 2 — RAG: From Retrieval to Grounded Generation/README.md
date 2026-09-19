RAG — From Retrieval to Grounded Generation
A Retrieval-Augmented Generation (RAG) system that retrieves relevant information from documents and uses an LLM to generate context-aware answers based on the retrieved content.

How It Works:
User Query -> Query Embedding -> FAISS Similarity Search -> Top-K Relevant Chunks -> Context Construction -> Prompt Creation -> LLM -> Grounded Answer

Tech Stack:
1. Python
2. Sentence Transformers
3. FAISS
4. NumPy
5. PDF/Text Processing
6. Large Language Model (LLM)

Key Concepts:
1. Retrieval-Augmented Generation (RAG)
2. Context construction
3. Prompt engineering for RAG
4. Grounded response generation
5. LLM integration
6. Source attribution
7. Hallucination prevention
8. Error handling and response extraction

Goal: To extend the semantic document search system into a RAG-based question-answering pipeline that combines document retrieval with LLM generation to produce relevant and context-grounded responses.

Future Scope: Advanced RAG, hybrid search, reranking, retrieval evaluation, query rewriting, conversational memory, and production-ready RAG deployment.
