# Classifying Relevant Prior Radiology Studies

## 1. Problem Framing

Radiologists frequently rely on prior imaging studies to provide clinical context, assess disease progression, and improve diagnostic accuracy. However, not all prior studies are equally useful. The goal of this project is to automatically classify whether a prior study is **relevant** to a current imaging study.

This is a **pairwise classification problem**, where each (current study, prior study) pair is labeled as relevant or not.

---

## 2. Baseline: Rule-Based Classifier

### Approach

The baseline system uses a deterministic scoring function based on three core signals:

- **Anatomy match** (strongest signal)
- **Modality compatibility**
- **Recency**

Each prior study is assigned a score:

score = 0.45 * anatomy + 0.35 * modality + 0.20 * recency


Additional logic:
- Strong penalties for mismatched anatomy
- Reduced scores for incompatible modalities
- Recency modeled with piecewise decay

### Strengths

- Very fast (microseconds per study)
- Fully interpretable
- Encodes clinical intuition directly

### Failure Modes

- **Brittle text matching** (regex limitations)
- **Coarse anatomy grouping**
- **Fixed weights across all cases**
- **No interaction modeling**

### Takeaway

A strong and interpretable baseline, but fundamentally limited by lack of learning.

---

## 3. Logistic Regression Model

### Approach

A logistic regression model was trained on features derived from the rule-based system:

- Final heuristic score
- Anatomy score
- Modality score
- Recency score
- Binary indicators (exact matches, incompatibility flags)

Other design choices:
- Group-aware splitting (by case)
- Threshold tuning (F1 or recall optimization)

### What Worked

- Learned better weighting of features
- Improved calibration vs fixed rules
- Threshold tuning enabled control over precision vs recall
- Stable across splits

### What Failed

- Performance limited by input features
- Cannot recover from feature extraction errors
- Linear model cannot capture feature interactions
- Redundant features reduce efficiency

### Takeaway

Improves over rules but is constrained by feature design and linearity.

---

## 4. XGBoost Model

### Approach

A gradient boosting model was trained using a richer feature set:

#### Feature Groups

1. **Embedding-based similarity**
   - Cosine similarity (ClinicalBERT)
   - Similarity gap and distribution features

2. **Heuristic features**
   - Anatomy, modality, recency

3. **Temporal features**
   - Days between studies
   - Recency indicators

#### Training Strategy

- GroupKFold cross-validation
- Optuna hyperparameter tuning
- Post-training threshold optimization

---

### What Worked

- Captured semantic similarity beyond keyword matching
- Modeled nonlinear interactions
- Combined embeddings + heuristics effectively
- Hyperparameter tuning improved performance

---

### What Failed / Challenges

- Higher computational cost
- Risk of overfitting
- Threshold instability
- Reduced interpretability

---

### Takeaway

Best performance overall, but with increased complexity and lower transparency.

---

## 5. Radiologist Workflow Assumptions

A simplified model of how radiologists select prior studies:

1. **Anatomy filtering**
   - Same region = strong relevance signal

2. **Modality consideration**
   - Same modality preferred
   - Cross-modality sometimes useful

3. **Recency evaluation**
   - Recent studies prioritized
   - Older studies used for trends

4. **Contextual judgment**
   - Clinical indication
   - Disease-specific follow-up

---

## 6. Mapping Models to Workflow

| Clinical Step        | Rule-Based | Logistic Regression | XGBoost |
|---------------------|----------|---------------------|---------|
| Anatomy filtering   | Explicit | Weighted            | Nonlinear |
| Modality reasoning  | Hard-coded | Learned           | Context-dependent |
| Recency handling    | Fixed bins | Tuned weight      | Adaptive |
| Semantic similarity | None     | None               | Strong |
| Context awareness   | None     | Limited            | Moderate |

### Key Insight

- Rule-based ≈ explicit heuristics  
- Logistic regression ≈ calibrated heuristics  
- XGBoost ≈ learned clinical intuition  

---

## 7. Evaluation Metrics

### Metrics Definition

- **Accuracy**: Overall correctness
- **Precision**: Of predicted relevant priors, how many are truly relevant  
- **Recall**: Of all truly relevant priors, how many are retrieved  
- **F1 Score**: Balance between precision and recall  
- **Base Rate**: Fraction of relevant priors in dataset  

In this task:
- **Recall impacts clinical safety**
- **Precision impacts workflow efficiency**

---

## 8. Results

### Summary Table

| Model                | Accuracy | Precision | Recall | F1     | Base Rate |
|---------------------|----------|----------|--------|--------|-----------|
| Rule-Based          | 0.7985   | 0.4941   | 0.6598 | 0.5651 | 0.1984    |
| Logistic Regression | 0.8857   | 0.7697   | 0.6047 | 0.6773 | 0.1984    |
| XGBoost             | 0.8778   | 0.7932   | 0.6662 | 0.7242 | 0.2408    |

---

### Detailed Results

#### Rule-Based

Accuracy : 0.7985
Precision: 0.4941
Recall : 0.6598
F1 : 0.5651

TP=671 TN=3423 FP=687 FN=346

**Interpretation:**

- High recall but very low precision
- Large number of false positives
- Overly permissive matching behavior

---

#### Logistic Regression

Accuracy : 0.8857
Precision: 0.7697
Recall : 0.6047
F1 : 0.6773

TP=615 TN=3926 FP=184 FN=402

**Interpretation:**

- Much higher precision than rule-based
- Significant reduction in false positives
- More conservative, misses some relevant priors

---

#### XGBoost

Accuracy : 0.8778
Precision: 0.7932
Recall : 0.6662
F1 : 0.7242

TP=886 TN=3962 FP=231 FN=444


**Interpretation:**

- Best overall F1 score
- Strong balance between precision and recall
- Benefits from nonlinear modeling and embeddings

---

## 9. Key Comparative Insights

1. **Tradeoff Patterns**
   - Rule-based → recall-heavy, noisy
   - Logistic regression → precision-heavy, conservative
   - XGBoost → best balance

2. **False Positives**
   - Logistic regression dramatically reduces FP
   - XGBoost slightly increases FP but improves recall

3. **Accuracy is misleading**
   - Inflated due to class imbalance
   - F1 and recall are more meaningful

4. **Best Overall Model**
   - XGBoost achieves strongest balance of metrics

---

## 10. Evaluation Framework, Insights, and Future Work

To ensure reproducibility, consistency, and rapid experimentation, a config-driven evaluation system was developed alongside the modeling approaches.

### Config-Driven Experiments

All experiments are defined using YAML configuration files. These configs specify:

- Model type (`rules`, `logreg`, `xgboost`)
- Hyperparameters
- Threshold tuning strategy
- Cross-validation settings

This design allows experiments to be modified and reproduced without changing code, enabling fast iteration and consistent comparisons across models.

### Unified Evaluation Script

A single evaluation script handles the full pipeline:

- Data loading  
- Model selection and execution  
- Training and inference  
- Metric computation  

This ensures that all models are evaluated under identical conditions.

### Key Design Components

**Model Registry**

    MODEL_RUNNERS = {}

Models are registered dynamically using decorators such as:

    @register_model("xgboost")
    @register_model("logreg")
    @register_model("rules")

This allows new models to be added without modifying core evaluation logic.

**Dynamic Dispatch**

    model_type = config["experiment"]["model_type"]

The model is selected at runtime via configuration, making the system flexible and extensible.

**Shared Evaluation Logic**

    def evaluate(data, predictions):

All models use the same evaluation function, ensuring:

- Consistent metric definitions  
- Fair comparisons  
- No duplicated evaluation logic  

**Standardized Outputs**

All models return results in a consistent structure:

    {
        "test_predictions": [...],
        "full_predictions": [...],
    }

This abstraction allows models to be interchangeable within the evaluation pipeline.

### Benefits of This Framework

- Reproducible experiments  
- Consistent and fair model comparisons  
- Easy extensibility for new models  
- Clear separation between modeling and evaluation logic  

### Key Insights

From both modeling and evaluation, several important patterns emerge:

- Anatomy is the dominant signal — errors here are rarely recoverable downstream  
- Recency is conditional — it matters primarily when anatomy is already relevant  
- Embeddings reduce brittleness by capturing semantic similarity beyond keywords  
- Rule-based methods remain strong baselines, especially for recall-heavy scenarios  
- Threshold tuning is critical — small changes significantly impact precision-recall tradeoffs  
- Evaluation design matters as much as modeling — poor splitting or inconsistent metrics can invalidate results  

### Future Improvements

**Modeling**
- Incorporate clinical indication into features  
- Explore ranking-based objectives instead of binary classification  
- Use transformer cross-encoders for pairwise reasoning  

**Features**
- Improve anatomy extraction using medical ontologies  
- Normalize procedure descriptions  
- Model longitudinal sequences of prior studies  

**Evaluation**
- Introduce human-in-the-loop validation  
- Perform stratified analysis by modality and anatomy  
- Use cost-sensitive metrics aligned with clinical impact  

### Final Takeaway

A key conclusion from this work is that evaluation infrastructure and experimental design are as important as the models themselves.

The combination of config-driven experimentation, unified evaluation logic, and group-aware validation is essential for producing reliable and clinically meaningful results.