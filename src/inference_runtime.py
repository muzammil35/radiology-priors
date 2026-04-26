# src/inference_runtime.py

from src.clinical_bert import ClinicalBERTEncoder, CaseEncoder

class InferenceRuntime:
    def __init__(self):
        self.encoder = ClinicalBERTEncoder("emilyalsentzer/Bio_ClinicalBERT")
        self.case_encoder = CaseEncoder(self.encoder)

runtime = InferenceRuntime()