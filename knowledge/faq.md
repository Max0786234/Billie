# Frequently Asked Questions

## Why is the first answer slow?

The language model has to be loaded into memory the first time it is used, which can take ten seconds or more on a laptop without a dedicated graphics card. Later answers are faster. Choosing a smaller model in Settings, or a cloud provider, also makes answers faster.

## How do I add my own documents?

Open the Documents panel in General mode and upload a PDF, Markdown or text file. Billie indexes it within a few seconds and you can then ask questions about it. Delete a document from the same panel to remove it from the knowledge base.

## How do I use a cloud model such as Claude or GPT?

Open Settings, choose a provider, paste your API key and pick a model. Use "Test connection" to check that the key works. Your choice is saved and used for all new answers. Switch back to Ollama at any time to work fully offline.

## Which local models are recommended?

On a laptop without a GPU, a 3 to 4 billion parameter model gives the best balance of speed and quality. Billie's default chat model is `qwen3:4b-instruct`, and `gemma3:4b` is used to read images. Larger models such as an 8 billion parameter model give better answers but are roughly twice as slow on a CPU.

## Why did Billie say it could not find the answer?

Billie only answers from its documents. If a question is outside the knowledge base, or uses very different words, retrieval may not find a relevant passage. Try rephrasing the question, switch to the right mode, or upload a document that contains the information.

## Does Billie support languages other than English?

Yes. Write in your language and Billie replies in the same language. The multilingual embedding model lets a question in Hindi, Bengali or Spanish find relevant English documents.
