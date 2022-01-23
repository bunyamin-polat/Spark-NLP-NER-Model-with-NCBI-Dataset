
import streamlit as st
import pandas as pd
import base64
import os

import sparknlp
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession

from sparknlp.annotator import *
from sparknlp.base import *

from sparknlp_display import NerVisualizer
from sparknlp.base import LightPipeline

spark = sparknlp.start(gpu = True) 


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.sidebar.image('https://nlp.johnsnowlabs.com/assets/images/logo.png', use_column_width=True)
st.sidebar.header('Choose the pretrained model')
select_model = st.sidebar.selectbox("",["ner_model_glove_100d"])

st.title("Spark NLP NER Model Playground")

#data
text1 = """The patient is a 78-year-old gentleman with no substantial past medical history except for diabetes. He denies any comorbid complications of the diabetes including kidney disease, heart disease, stroke, vision loss, or neuropathy. At this time, he has been admitted for anemia with hemoglobin of 7.1 and requiring transfusion. He reports that he has no signs or symptom of bleeding and had a blood transfusion approximately two months ago and actually several weeks before that blood transfusion, he had a transfusion for anemia. He has been placed on B12, oral iron, and Procrit. At this time, we are asked to evaluate him for further causes and treatment for his anemia. He denies any constitutional complaints except for fatigue, malaise, and some dyspnea. He has no adenopathy that he reports. No fevers, night sweats, bone pain, rash, arthralgias, or myalgias."""
text2 = """The patient is a 61-year-old woman who presents with a history of biopsy-proven basal cell carcinoma, right and left cheek. She had no prior history of skin cancer. She is status post bilateral cosmetic breast augmentation many years ago and the records are not available for this procedure. She has noted progressive hardening and distortion of the implant. She desires to have the implants removed, capsulectomy and replacement of implants. She would like to go slightly smaller than her current size as she has ptosis going with a smaller implant combined with capsulectomy will result in worsening of her ptosis. She may require a lift. She is not consenting to lift due to the surgical scars."""
text3 = """The patient is a 39-year-old woman returns for followup management of type 1 diabetes mellitus. Her last visit was approximately 4 months ago. Since that time, the patient states her health had been good and her glycemic control had been good, however, within the past 2 weeks she had a pump malfunction, had to get a new pump and was not certain of her pump settings and has been having some difficulty with glycemic control over the past 2 weeks. She is not reporting any severe hypoglycemic events, but is having some difficulty with hyperglycemia both fasting and postprandial. She is not reporting polyuria, polydipsia or polyphagia. She is not exercising at this point and has a diet that is rather typical of woman with twins and a young single child as well. She is working on a full-time basis and so eats on the run a lot, probably eats more than she should and not making the best choices, little time for physical activity. She is keeping up with all her other appointments and has recently had a good eye examination. She had lab work done at her previous visit and this revealed persistent hyperlipidemic state with a LDL of 144."""
text4 = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting ."""
text5 = """Nature and course of the diagnosis has been discussed with the patient. Based on her presentation without any history of obvious fall or trauma and past history of malignant melanoma, this appears to be a pathological fracture of the left proximal hip. At the present time, I would recommend obtaining a bone scan and repeat x-rays, which will include AP pelvis, femur, hip including knee. She denies any pain elsewhere. She does have a past history of back pain and sciatica, but at the present time, this appears to be a metastatic bone lesion with pathological fracture. I have discussed the case with Dr.X and recommended oncology consultation. With the above fracture and presentation, she needs a left hip hemiarthroplasty versus calcar hemiarthroplasty, cemented type. Indication, risk, and benefits of left hip hemiarthroplasty has been discussed with the patient, which includes, but not limited to bleeding, infection, nerve injury, blood vessel injury, dislocation early and late, persistent pain, leg length dicrepancy, myositis ossificans, intraoperative fracture, prosthetic fracture, need for conversion to total hip replacement surgery, revision surgery, pulmonary embolism, risk of anesthesia, need for blood transfusion, and cardiac arrest. She understands above and is willing to undergo further procedure. The goal and the functional outcome have been explained. Further plan will be discussed with her once we obtain the bone scan and the radiographic studies. We will also await for the oncology feedback and clearance."""

sample_text = st.selectbox("",[text1, text2, text3,text4,text5])

@st.cache(hash_funcs={"_thread.RLock": lambda _: None},allow_output_mutation=True, suppress_st_warning=True)
def model_pipeline():
    documentAssembler = DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

    sentenceDetector = SentenceDetector()\
          .setInputCols(['document'])\
          .setOutputCol('sentence')

    tokenizer = Tokenizer()\
          .setInputCols(['sentence'])\
          .setOutputCol('token')

    gloveEmbeddings = WordEmbeddingsModel.pretrained()\
          .setInputCols(["document", "token"])\
          .setOutputCol("embeddings")

    nerModel = NerDLModel.load("/content/drive/MyDrive/SparkNLPTask/NER_glove_100d_e14_b10")\
          .setInputCols(["sentence", "token", "embeddings"])\
          .setOutputCol("ner")

    nerConverter = NerConverter()\
          .setInputCols(["document", "token", "ner"])\
          .setOutputCol("ner_chunk")
 
    pipeline_dict = {
          "documentAssembler":documentAssembler,
          "sentenceDetector":sentenceDetector,
          "tokenizer":tokenizer,
          "gloveEmbeddings":gloveEmbeddings,
          "nerModel":nerModel,
          "nerConverter":nerConverter
    }
    return pipeline_dict

model_dict = model_pipeline()

@st.cache(hash_funcs={"_thread.RLock": lambda _: None},allow_output_mutation=True, suppress_st_warning=True)
def load_pipeline():
    nlp_pipeline = Pipeline(stages=[
                   model_dict["documentAssembler"],
                   model_dict["sentenceDetector"],
                   model_dict["tokenizer"],
                   model_dict["gloveEmbeddings"],
                   model_dict["nerModel"],
                   model_dict["nerConverter"]
                   ])

    empty_data = spark.createDataFrame([['']]).toDF("text")

    model = nlp_pipeline.fit(empty_data)

    return model


ner_model = load_pipeline()

def viz (annotated_text, chunk_col):
  raw_html = NerVisualizer().display(annotated_text, chunk_col, return_html=True)
  sti = raw_html.find('<style>')
  ste = raw_html.find('</style>')+8
  st.markdown(raw_html[sti:ste], unsafe_allow_html=True)
  st.write(HTML_WRAPPER.format(raw_html[ste:]), unsafe_allow_html=True)


def get_entities (ner_pipeline, text):
    
    light_model = LightPipeline(ner_pipeline)

    full_annotated_text = light_model.fullAnnotate(text)[0]

    st.write('')
    st.subheader('Entities')

    chunks=[]
    entities=[]
    
    for n in full_annotated_text["ner_chunk"]:

        chunks.append(n.result)
        entities.append(n.metadata['entity'])

    df = pd.DataFrame({"chunks":chunks, "entities":entities}).drop_duplicates()

    viz (full_annotated_text, "ner_chunk")
    
    st.subheader("Dataframe")
    st.write('')

    st.table(df)
    
    return df


entities_df  = get_entities (ner_model, sample_text)

