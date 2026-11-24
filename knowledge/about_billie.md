# About Billie

Billie is a retrieval-augmented AI assistant. Instead of answering only from what a language model memorised during training, Billie first searches a document collection, then asks the model to answer using the passages it found and to cite them with numbered references such as [1] and [2]. If the documents do not contain the answer, Billie is instructed to say so rather than guess.

## Modes

Billie has three modes, each backed by its own searchable corpus.

- **General** answers questions about the documents in Billie's knowledge base. This includes these help pages and any files you upload in the Documents panel (PDF, Markdown or plain text).
- **Medical** answers health questions from MedQuAD, a collection of about 16,400 question-answer pairs written by trusted U.S. National Institutes of Health sources such as MedlinePlus, Genetics Home Reference, GARD, NIDDK and the National Cancer Institute. Medical answers are educational information, not a diagnosis or a substitute for a clinician.
- **Research** helps you explore computer-science research papers from arXiv. It searches paper titles and abstracts and can summarise and compare papers, linking each one to its arXiv page.

## What Billie can do in every mode

- Hold a conversation: follow-up questions such as "what about its side effects?" are understood in the context of earlier turns in the same chat.
- Read images: you can attach a photo, screenshot or diagram and ask a question about it. Billie uses a vision-capable model to describe the image and answers using that description together with retrieved documents.
- Reply in your language: Billie detects whether you write in English, Hindi, Bengali, Spanish or many other languages and answers in the same language, even though most of its documents are in English.
- Adapt its tone: Billie detects frustration or worry in a message and responds with more care and clarity.

## Citations

Every answer lists the sources it used as cards under the message. A number in square brackets in the answer, such as [2], refers to the second source card. Click a card to read the full passage and open the original page.
