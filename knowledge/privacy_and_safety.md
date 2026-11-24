# Privacy and Safety

## Where your data goes

With the default local setup, questions, uploaded documents, images and chat history are processed on your own computer and stored in Billie's `data` folder. Nothing is sent to the internet.

If you choose a cloud model in Settings, the text of your question, the retrieved passages and any attached image are sent to that provider to generate the answer. API keys you enter are stored locally in `data/llm_settings.json` and are never shown back in full by the interface.

## Medical information

Medical mode provides general educational information drawn from MedQuAD, which was compiled from U.S. National Institutes of Health websites. It cannot examine you, does not know your medical history, and can be incomplete or out of date. Always consult a qualified healthcare professional for diagnosis or treatment. In an emergency, contact your local emergency number immediately.

## Limits of the assistant

Billie can still make mistakes. A citation shows where information came from, but the summary of that information can be imperfect, so check the cited passage for anything important. When the knowledge base does not contain the answer, Billie should say that it does not know instead of inventing one.

## Uploads

Images up to 8 MB (PNG, JPEG, WebP or GIF) and documents up to 20 MB (PDF, Markdown or text) are accepted. Files are checked by their content, not only by their name, before they are processed.
