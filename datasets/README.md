# Datasets

| Folder | In git | What it is | How to get it |
|---|---|---|---|
| `samples/` | yes | Tiny MedQuAD and arXiv samples, used by tests and as a fallback when the full indexes are not built | included |
| `medquad/` | no | Full MedQuAD corpus: 11k XML files, 16,407 answered QA pairs (CC BY 4.0) | `git clone https://github.com/abachaa/MedQuAD datasets/medquad` |
| `medquad/liveqa/` | no | MedQuAD's judged answers for the TREC-2017 LiveQA medical test questions, used by the retrieval benchmark | unzip `medquad/QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip` into `medquad/liveqa/` |
| `arxiv/` | no | Kaggle arXiv metadata snapshot, `arxiv-metadata-oai-snapshot.json` (~5.4 GB) | https://www.kaggle.com/datasets/Cornell-University/arxiv |

The raw files are read only by the index builder (`python -m billie.ingest`). The app serves
from the compact indexes it writes to `data/index/`.
