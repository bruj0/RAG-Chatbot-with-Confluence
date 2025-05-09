# Demo
if __name__ == '__main__':
    from help_desk import HelpDesk

    print("Using Ollama with Mistral model")
    
    model = HelpDesk(new_db=True)

    print(f"Vectorstore contains {model.db._collection.count()} document chunks")

    prompt = 'What is platform engineering?'
    print(f"\nQuestion: {prompt}")
    result, sources = model.retrieval_qa_inference(prompt)
    print("\nAnswer:", result)
    print("\nSources:", sources)
