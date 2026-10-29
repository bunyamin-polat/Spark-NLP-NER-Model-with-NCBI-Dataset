# Disease NER with Spark NLP on the NCBI Disease Corpus

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bunyamin-polat/Spark-NLP-NER-Model-with-NCBI-Dataset/blob/main/NER_Model_Training.ipynb)

A named entity recognition (NER) model that finds **disease mentions** in biomedical and clinical text. It is trained with Spark NLP's deep-learning NER annotator on the NCBI Disease corpus and served through a small Streamlit demo.

Given the sentence *"…a history of gestational diabetes mellitus … one prior episode of HTG-induced pancreatitis…"*, the model returns entities such as `gestational diabetes mellitus`, `HTG-induced pancreatitis`, `acute hepatitis` and `obesity`, each labelled `Disease`.

## Results

Token-level evaluation on the official NCBI Disease **test set** (940 sentences, 24,497 tokens):

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `B-Disease` (first token of a disease mention) | 0.86 | 0.85 | **0.85** | 960 |
| `I-Disease` (continuation tokens) | 0.80 | 0.89 | **0.84** | 1,087 |
| `O` (outside) | 0.99 | 0.99 | 0.99 | 22,450 |

On the training set the same model reaches an F1 of 0.96 on both disease labels. The gap of about 0.11 to the test set shows moderate overfitting. Full classification reports and the per-epoch training log are in the notebook.

## Approach

| Component | Choice |
| --- | --- |
| Framework | Spark NLP 3.4.0 on Apache Spark 3.1.2 (Google Colab, GPU) |
| Annotator | `NerDLApproach`: a BiLSTM-CNN-Char network |
| Word embeddings | Pretrained `glove_100d` |
| Hyperparameters | 14 epochs, learning rate 0.003, batch size 10, dropout 0.5, 20% validation split, fixed random seed |
| Tagging scheme | IOB with a single entity type, `Disease` |
| Training time | About 59 minutes |

The trained model is used in a prediction pipeline: `DocumentAssembler → SentenceDetector → Tokenizer → GloVe embeddings → NerDLModel → NerConverter`. A `LightPipeline` serves fast predictions on single documents.

## Data

**NCBI Disease corpus:** 793 PubMed abstracts in which disease mentions were annotated by domain experts.

| Split | Sentences | `B-Disease` | `I-Disease` | `O` |
| --- | ---: | ---: | ---: | ---: |
| Train + development | 6,347 | 5,921 | 7,205 | 146,544 |
| Test | 940 | 960 | 1,087 | 22,450 |

[`Prepare_CoNLL_Data_Format.ipynb`](Prepare_CoNLL_Data_Format.ipynb) converts the IOB version published in [cambridgeltl/MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) into the CoNLL format that Spark NLP reads. The resulting files are included: `NCBI-disease-IOB_train_devel.conll` and `NCBI-disease-IOB_test.conll`.

> Doğan, R. I., Leaman, R., & Lu, Z. (2014). NCBI disease corpus: A resource for disease name recognition and concept normalization. *Journal of Biomedical Informatics*, 47, 1–10.

## Repository Contents

| File | Purpose |
| --- | --- |
| [`NER_Model_Training.ipynb`](NER_Model_Training.ipynb) | Data loading, label analysis, training, evaluation, prediction pipeline, Streamlit launch |
| [`Prepare_CoNLL_Data_Format.ipynb`](Prepare_CoNLL_Data_Format.ipynb) | Converts the IOB data to CoNLL |
| [`ner_model_streamlit.py`](ner_model_streamlit.py) | Streamlit app that highlights disease entities in sample clinical notes |
| `*.conll` | Training and test data in CoNLL format |

## How to Run

1. Open `NER_Model_Training.ipynb` in Google Colab with a GPU runtime and run all cells. The notebook downloads the data, trains the model and saves it as `NER_glove_100d_e14_b10`.
2. **Streamlit demo (optional):**
   - Set the path in `NerDLModel.load(...)` in `ner_model_streamlit.py` to the folder of the saved model.
   - Set your own ngrok token in the `NGROK_AUTHTOKEN` environment variable.
   - Run the last cells of the notebook. They start the app and print a public ngrok URL.

## Limitations

- **Token-level metrics only.** Precision and recall are computed per token, not per complete entity (exact span match), so they are more optimistic than entity-level scores.
- **One training run with general-domain embeddings.** There was no hyperparameter search and no comparison with biomedical embeddings, such as clinical or BioBERT embeddings, which typically perform better on this task.
- **2022 software versions.** The notebook pins Spark 3.1.2 and Spark NLP 3.4.0; newer releases may need small API changes.
- The model detects one entity type (`Disease`) and does not normalise mentions to an ontology such as MeSH.

## License

The code and notebooks are released under the [MIT License](LICENSE). The NCBI Disease corpus files are not covered by this license; they remain under the terms of the original corpus (see Data).
