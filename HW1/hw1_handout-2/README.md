# Language Classification using N-gram Model

This project trains an N-gram model to classify six different languages by analyzing character frequency instead of whole words.

## Goal
- Train an N-gram model for six-language classification.
- Use character frequency for language classification rather than full words.

## Experiment
- Run the HW1.ipynb

## Data
- train.tsv and test.tsv
- Format: `(language, sentence)`
- Example:
  - Swahili: `Unatakiwa ukumbuke kuwa ulikuwa mtumwa katika nchi ya Misri; kwa hiyo nakuagiza kutii amri hii.`
  - Manobo: `Na impananglitan ni Jesus to mgo Hudiyu bahin to sinakupan din no iyan ogka-angodan to pigpang-aad no mgo buhì no karniru.`

## Model
- Multinomial Logistic Regression.

## Evaluation
- Metrics: Precision, Recall, F1 Score.
- Multiclass Classification:
  - Micro-Averaged
  - Macro-Averaged

## Data Preprocessing

### 1. Feature Extraction: Extract N-grams
- Use list comprehension: `[x[i:] for i in range(n)]`
  - Example:
    - For `x = "abcdef"` and `n = 3`:
      - i=0: "abcdef"
      - i=1: "bcdef"
      - i=2: "cdef"
    - Result: `["abcdef", "bcdef", "cdef"]`
- Zip results: `zip(*[x[i:] for i in range(n)])`
  - Produces tuples like: `('a', 'b', 'c'), ('b', 'c', 'd'), ...`
- Join N-grams: `["abc", "bcd", "cde", "def"]`

### 2. Language Codes to One-Hot Vector
- Example:
  - `langs = ["English", "French", "Spanish"]`
  - For `lang = "French"`, `langs.index("French")` returns `1`.
  - Resulting vector: `[0, 1, 0]`

### 3. Vectorize N-grams
- `Counter`: Count occurrences of each N-gram, e.g., `{"abc": 0, "bcd": 2, "cde": 5 ...}`
- Feature Map: Maps N-grams to indices, e.g., `{"abc": 0, "bcd": 1, "cde": 2}`
- Feature Vector: `[0, 2, 5, …]`

### 4. Combine Preprocessing Steps for Observations
- Example:
  - Training observations: `[("English", "abcdef"), ("French", "bcdefg")]`
  - Feature map: `{"abc": 0, "bcd": 1, "cde": 2, "def": 3, "efg": 4}`
  - Language map: `{"English": [1, 0], "French": [0, 1]}`
  - Observation: `[([1, 0], [1, 1, 1, 1, 0]), ([0, 1], [0, 1, 1, 1, 1])]`

## Training and Evaluation

1. **Gradient Descent**:
   - Optimize the model parameters.

2. **Classification**:
   - Compute: `softmax(W @ x)`
   - Select the language with the highest probability.

3. **Evaluation Metrics**:
   - Track `true positives`, `false positives`, and `false negatives`.

4. **Prediction**:
   - Returns the predicted language name.

5. **Confusion Matrix**:
   - A `k x k` matrix, e.g.:
     ```plaintext
     [[18, 1, 0, 0, 9, 0],
      [0, 31, 0, 0, 0, 1],
      [0, 0, 33, 0, 0, 1],
      [0, 1, 0, 34, 1, 0],
      [0, 0, 0, 0, 142, 0],
      [0, 0, 0, 0, 1, 120]]
     ```

6. **Misclassified Examples**:
   - Output misclassified sentences for review.

7. **Feature Weight Analysis**:
   - Print the top 10 most important features for each language.
   - Example function:
     ```python
     def most_important_feature_for_language(language: str):
         language_idx = np.argmax(lang_map[language])
         language_feature_vector = W_inspect[language_idx, :]
         top_indices = np.argsort(language_feature_vector)[::-1][:10]
         top_features = [(inverted_feature_map[idx], language_feature_vector[idx]) for idx in top_indices]
     ```
   - Example Maps:
     - `lang_map = {"English": [1, 0, 0], "French": [0, 1, 0], "Spanish": [0, 0, 1]}`
     - `inverted_feature_map = {0: "the", 1: "un", 2: "le", 3: "y", ...}`
     - `W_inspect` contains weights per feature for each language.
