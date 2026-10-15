# SOPilot AI — Manufacturing SOP Assistant

## 1. Project Overview

SOPilot AI is an educational prototype that helps users search sample manufacturing procedures and workplace safety reference documents. It combines a fine-tuned Small Language Model (SLM), semantic retrieval, and Retrieval-Augmented Generation (RAG).

The application retrieves potentially relevant information and uses it as context for generated answers. It also supports model comparison, application logging, and user feedback.

**Important:** SOPilot AI is not a certified workplace-safety system. Its sample procedures are not verified company policies, and public safety references are not substitutes for employer-approved SOPs.

## 2. Key Features

- **Semantic retrieval:** Uses sentence embeddings to identify potentially relevant passages.
- **Fine-tuned SLM:** Uses a QLoRA adapter for Qwen2.5-0.5B-Instruct.
- **RAG-based responses:** Supplies retrieved reference passages to the language model.
- **Model comparison:** Compares base-model, fine-tuned-model, and retrieval-assisted responses.
- **Safety safeguards:** Includes targeted checks for certain unsupported company-specific questions.
- **Logging and monitoring:** Records application activity and user feedback.
- **Streamlit interface:** Provides a user interface for interacting with the assistant.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Language model | Qwen/Qwen2.5-0.5B-Instruct |
| Fine-tuning | QLoRA, LoRA, PEFT |
| Model framework | PyTorch, Hugging Face Transformers |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Retrieval | Sentence embeddings and cosine similarity |
| Application interface | Streamlit |
| Training environment | Google Colab with NVIDIA Tesla T4 GPU |
| Development environment | Visual Studio Code on Windows |
| Evaluation output | CSV files |
| Logging | Python logging and CSV files |

## 4. Project Architecture

The application follows this general workflow:

1. **User input:** The user submits a question through the Streamlit interface.
2. **Procedure retrieval:** The assistant searches the available sample procedure records.
3. **Semantic retrieval:** The question is compared with embedded passages from the available OSHA confined-space reference.
4. **Context preparation:** Retrieved information is assembled for the response-generation stage.
5. **Answer generation:** The fine-tuned Qwen model generates a response using the question and available context.
6. **Safety checks:** Targeted checks help prevent unsupported answers to selected company-specific questions.
7. **Logging and feedback:** Application activity and user feedback are recorded for review.
8. **Evaluation:** Separate scripts compare retrieval methods and model responses.

Retrieval and safety checks reduce some risks but do not guarantee that every generated answer is accurate or supported by the source documents.

## 5. Data Sources

### Sample procedure dataset

The project uses a JSON dataset containing 50 sample safety procedures. These records support demonstration and retrieval testing.

The sample records should not be treated as verified company-approved procedures.

### Public safety reference

The project uses an OSHA confined-space safety reference document:

- Source file: `osha_confined_space_quick_card.pdf`
- Processed chunks: `data/processed/osha_confined_space_chunks.json`

The extracted passages are used for semantic retrieval. Their relevance and completeness must be checked against the original reference.

### Fine-tuning dataset

The fine-tuning dataset contains 100 instruction-and-response examples prepared for supervised fine-tuning.

This small dataset is intended for prototyping and does not establish comprehensive workplace-safety knowledge.

## 6. Fine-Tuning

The base model is `Qwen/Qwen2.5-0.5B-Instruct`.

The model was fine-tuned using QLoRA in Google Colab with an NVIDIA Tesla T4 GPU. QLoRA uses quantization and trainable low-rank adapters to reduce the resources required for fine-tuning.

The saved adapter is located at:

`models/sopilot_qlora_adapter/`

The application loads the base model and the fine-tuned adapter for inference.

The exact training hyperparameters, training duration, and resource usage should be taken from the original training notebook if they are required for reproducibility. They should not be estimated or invented.

## 7. Retrieval and RAG

The embedding model is:

`sentence-transformers/all-MiniLM-L6-v2`

The semantic retrieval workflow:

1. Loads the processed document chunks.
2. Encodes the user's question and the document passages into embeddings.
3. Calculates cosine similarity between the question and passages.
4. Ranks candidate passages and returns the selected results.
5. Supplies the retrieved information to the model as context.

The current implementation includes a similarity threshold and a keyword boost for atmospheric-testing questions.

**Known limitation:** A high similarity score does not guarantee that a passage answers the question. The system may retrieve irrelevant information, and the language model may still generate unsupported statements.

## 8. Evaluation

Evaluation is preliminary and intended to identify areas for improvement. It is not a formal safety validation.

### Semantic retrieval evaluation



- **Test questions:** 10
- **Correct top-ranked results:** 10
- **Top-1 title-match rate:** 100%
- **Limitation:** This was a small prototype test and does not establish safety or operational reliability.

The evaluation indicates that semantic retrieval ranked the expected procedure first for all 10 test questions. Further testing with a larger, independently verified question set is needed.

### TF-IDF versus semantic retrieval

A separate preliminary comparison tested eight questions using TF-IDF and semantic retrieval.

The tests covered confined-space hazards, atmospheric testing, PPE, permits, communication, company-specific information, and an unrelated conveyor-belt repair question.

The results showed that semantic retrieval returned matches for some questions whose answers were not actually available in the documents. The unrelated conveyor-belt question returned no match in either method.

These results demonstrate why retrieval relevance must be reviewed separately from whether a result was returned. The two methods' scores are not directly comparable.

### Base model versus fine-tuned model versus RAG

The comparison script is:

`compare_models.py`

It tests the same 12 questions against:

- Base Qwen model
- Fine-tuned Qwen model
- Fine-tuned Qwen model with retrieved OSHA context

The output is saved to:

`results/model_comparison.csv`

The comparison identified useful responses for some confined-space questions, but also revealed unsupported or irrelevant answers. For example, retrieval-assisted answers to company-specific evacuation and contact-number questions were not adequately grounded in the available reference material.

The comparison script uses retrieved OSHA context and does not reproduce every step of the complete Streamlit application's retrieval pipeline. Its results should therefore be interpreted as a preliminary model comparison, not as a definitive ranking of application performance.

### Evaluation limitations

- The evaluation sets are small.
- Keyword matches are only a proxy for retrieval relevance.
- Retrieved context may be unrelated to the question.
- Generated answers may omit important information or introduce unsupported claims.
- No formal independent safety assessment has been performed.
- The results do not establish production readiness or real-world safety accuracy.

## 9. Logging and Monitoring

The application includes logging and feedback collection.

Relevant files include:

- `data/app_logs.csv` — application activity logs.
- `data/feedback_log.csv` — user feedback records.

These files support basic review of application behavior and user feedback. The logging implementation should be inspected before making claims about the exact events or fields recorded.

Logs may contain user questions or other operational information. Avoid recording confidential workplace details or unnecessary personal information.

## 10. Installation and Setup

### Prerequisites

- Python installed on Windows.
- Visual Studio Code.
- The SOPilot AI project files.
- The saved fine-tuned adapter.
- The required data files.
- Internet access for downloading model and embedding files when they are not already cached.

### Set up the project

1. Open the `SOPilotAI` folder in Visual Studio Code.
2. Open **Terminal → New Terminal**.
3. Activate the project's existing virtual environment, if applicable.
4. Ensure the required Python dependencies are installed.
5. Verify that the adapter and processed data files are present.

Do not overwrite the existing virtual environment or reinstall packages unnecessarily if the application is already working.

### Run the Streamlit application

From the project root, run:

`streamlit run app.py`

Open the local URL printed in the terminal.

The first model load may take time, especially on a CPU-only computer.

### Run model comparison

From the project root, run:

`python compare_models.py`

The script compares the base model, fine-tuned model, and fine-tuned model with retrieved context.

Results are saved to `results/model_comparison.csv`. The script may take several minutes on a CPU-only system.
### Model Comparison

The project compares three approaches using the same 11 test questions:

1. **Base model:** Qwen2.5-0.5B-Instruct without fine-tuning or retrieved context.
2. **Fine-tuned model:** The model with the SOPilot QLoRA adapter.
3. **RAG-assisted model:** The fine-tuned model supported by retrieved procedure and public-reference information.

**Evaluation output:** `results/model_comparison.csv`
### Evaluation Results

SOPilot AI was evaluated using a small set of 10 manufacturing safety questions.

| Metric | Result |
|---|---:|
| Questions manually scored | 10 |
| Average relevance score | 4.1/5 |
| Average safety score | 4.0/5 |
| Semantic retrieval test questions | 10 |
| Correct top-ranked retrieval results | 10/10 |
| Top-1 title-match rate | 100% |

**Evaluation files**
- `data/training/sopilot_evaluation_results.csv`
- `data/training/sopilot_evaluation_results_scored.csv`
- `data/training/retrieval_evaluation.txt`
- `results/model_comparison.csv`

The retrieval evaluation achieved a 100% top-1 title-match rate on its 10-question prototype test. The separate manual evaluation recorded average relevance and safety scores of 4.1/5 and 4.0/5, respectively.

These are preliminary results from small test sets. Manual scoring is subjective, and successful retrieval does not guarantee that a generated answer is correct. The system requires broader evaluation and validation before any real-world safety-critical use.

The CSV records the question, each approach's generated answer, and whether retrieved context was found.

**Preliminary observations:**
- The approaches produce different answers to the same questions.
- Retrieved context can provide relevant source information for confined-space questions.
- The system sometimes generates incomplete or unsupported answers, highlighting the need for further evaluation.
- For company-specific questions, such as evacuation routes and safety-officer contact details, the RAG approach appropriately indicates when the information cannot be verified.
- An out-of-scope conveyor motor repair question did not retrieve a relevant procedure.

**Limitation:** This is a small prototype comparison. Answer quality has not yet been established through a complete, standardized scoring evaluation.

### Run retrieval comparison

From the project root, run:

`python compare_retrieval.py`

This script compares TF-IDF and semantic retrieval for a small set of questions.

### Monitoring

The project includes logging-related files and monitoring functionality. Check the actual monitoring script and its configured log paths before running it. Do not assume that a log file exists at a different location from the application's configured path.

## 11. Project File Structure

The principal project files include:

```text
SOPilotAI/
├── app.py
├── sopilot_assistant.py
├── semantic_retrieval.py
├── retrieve_procedure.py
├── compare_models.py
├── compare_retrieval.py
├── evaluate_sopilot.py
├── model_config.py
├── load_finetuned_model.py
├── logging_config.py
├── models/
│   └── sopilot_qlora_adapter/
├── data/
│   ├── app_logs.csv
│   ├── feedback_log.csv
│   ├── processed/
│   │   └── osha_confined_space_chunks.json
│   └── training/
│       └── sopilot_evaluation_results.csv
└── results/
    └── model_comparison.csv
```

This is a list of principal files, not a complete inventory of every file in the project. Some supporting scripts and datasets may also be present.

## 12. Safety Disclaimer

SOPilot AI is an educational prototype and is not certified for operational workplace-safety use.

- Sample procedures are not verified company-approved SOPs.
- Public reference passages may be incomplete or irrelevant to a particular question.
- Generated answers may contain errors or unsupported statements.
- Targeted safeguards do not guarantee that all unsupported questions will be handled correctly.
- Never use this prototype as the sole basis for an emergency response or safety-critical decision.
- Always follow current, employer-approved procedures and consult qualified safety personnel.

## 13. Future Improvements

Potential improvements include:

1. Improve retrieval relevance for PPE, permits, communication, and other question categories.
2. Add a reliable answerability check so irrelevant passages are not treated as evidence.
3. Expand testing with independently reviewed questions and expected source passages.
4. Evaluate answer relevance, factual grounding, completeness, and unsupported claims.
5. Improve error handling and review persistent logs.
6. Add secure configuration and privacy controls before any external deployment.
7. Document training hyperparameters and measure performance using repeatable tests.
8. Test the application with authorized, organization-specific SOPs before considering operational use.

## 14. References

- Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-In