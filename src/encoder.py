from sklearn.preprocessing import LabelEncoder
from src.classifier import extract_anatomy, extract_modality, fallback_modality

class Encoders:
    def __init__(self):
        self.modality_encoder = LabelEncoder()
        self.anatomy_encoder = LabelEncoder()

    def fit(self, cases):
        modalities = set()
        anatomies = set()

        for case in cases:
            cur_mod = extract_modality(case.current_study.study_description)
            cur_anat = extract_anatomy(case.current_study.study_description)

            if cur_mod is None:
                cur_mod = fallback_modality(case.current_study.study_description, cur_anat)
    

            modalities.add(cur_mod)
            anatomies.add(cur_anat)

            for prior in case.prior_studies:
                pri_mod = extract_modality(prior.study_description)
                pri_anat = extract_anatomy(prior.study_description)
                if pri_mod is None:
                    pri_mod = fallback_modality(prior.study_description, pri_anat)
                    print(pri_mod)

                modalities.add(pri_mod)
                anatomies.add(pri_anat)

        self.modality_encoder.fit(list(modalities))
        self.anatomy_encoder.fit(list(anatomies))

    def transform(self, modality, anatomy):
        # handle None safely
        modality = modality if modality is not None else "UNKNOWN"
        anatomy = anatomy if anatomy is not None else "UNKNOWN"

        print(modality)
        print(anatomy)

        return (
            self.modality_encoder.transform([modality])[0],
            self.anatomy_encoder.transform([anatomy])[0],
        )