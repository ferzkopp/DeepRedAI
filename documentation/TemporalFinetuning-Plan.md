# Temporal Knowledge Cutoff Fine-Tuning

## Objective

Fine-tune a general knowledge language model to become "blind" to factual knowledge after a specific date (e.g., 1969), effectively creating a model that behaves as if it only has knowledge up to that temporal boundary.

---

## Temporal Dataset Generation

Creating reliable pools of pre-cutoff and post-cutoff knowledge is a **foundational prerequisite** for all fine-tuning approaches. This is an independent task that requires careful temporal classification of factual content.

### The Challenge

Not all facts have explicit dates. Many questions like "Who invented the lightbulb?" don't contain temporal markers but reference events with known dates. We need robust methods to:
1. Extract temporal information from text
2. Classify facts by their "knowledge date"
3. Handle ambiguous or timeless facts
4. Scale to hundreds of thousands of examples

---

### Source Datasets

| Dataset | Size | Description | Temporal Utility | Link |
|---------|------|-------------|------------------|------|
| **TriviaQA** | 650K+ Q&A pairs | Trivia questions with Wikipedia/web evidence | Many questions reference specific dates/events | [HuggingFace](https://huggingface.co/datasets/mandarjoshi/trivia_qa) |
| **Natural Questions (NQ)** | 307K examples | Real Google search queries with Wikipedia answers | Contains timestamped URLs, date-referenced events | [GitHub](https://github.com/google-research-datasets/natural-questions) |
| **SQuAD 1.1/2.0** | 100K+ Q&A pairs | Wikipedia-based reading comprehension | Context passages contain temporal information | [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad) |
| **MMLU** | 15K+ test questions | 57 subjects including history, science | Clear subject separation (world_history, us_history) | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| **HotpotQA** | 113K Q&A pairs | Multi-hop reasoning over Wikipedia | Questions chain temporal facts | [HuggingFace](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |
| **Jeopardy Dataset** | 200K+ clues | Historical trivia with date/era categories | Well-structured temporal categories | [Kaggle/GitHub](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/) |
| **WikiQA** | 3K questions | Wikipedia-sourced Q&A | Smaller scale, good for initial experiments | [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52419) |
| **TempQuestions** | 1.3K questions | Explicitly temporal questions with date annotations | Purpose-built for temporal reasoning | [Academic](https://www.aclweb.org/anthology/P18-2057/) |
| **Wikidata** | Billions of facts | Structured knowledge base with temporal properties | Direct date fields (P580/P582 for start/end dates) | [Wikidata](https://www.wikidata.org/) |
| **DBpedia** | Structured Wikipedia extracts | Ontology-based knowledge | Birth dates, event dates, founding dates | [DBpedia](https://www.dbpedia.org/) |

---

### ML-Assisted Temporal Classification

Manual classification doesn't scale. Here's how ML can help:

#### 1. Named Entity Recognition (NER) + Knowledge Base Linking

Extract entities and look up their temporal properties in knowledge bases:

```python
import spacy
from SPARQLWrapper import SPARQLWrapper, JSON

nlp = spacy.load("en_core_web_trf")  # Transformer-based for better accuracy

def get_wikidata_dates(entity_id):
    """Query Wikidata for temporal properties of an entity"""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?birthDate ?deathDate ?startDate ?endDate ?inceptionDate WHERE {{
      OPTIONAL {{ wd:{entity_id} wdt:P569 ?birthDate. }}   # birth date
      OPTIONAL {{ wd:{entity_id} wdt:P570 ?deathDate. }}   # death date  
      OPTIONAL {{ wd:{entity_id} wdt:P580 ?startDate. }}   # start time
      OPTIONAL {{ wd:{entity_id} wdt:P582 ?endDate. }}     # end time
      OPTIONAL {{ wd:{entity_id} wdt:P571 ?inceptionDate. }} # inception
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def extract_temporal_entities(text):
    """Extract entities and their temporal relevance"""
    doc = nlp(text)
    entities_with_dates = []
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "EVENT", "WORK_OF_ART", "GPE"]:
            # Link to Wikidata and get dates
            # (In practice, use entity linking like REL, BLINK, or mGENRE)
            entities_with_dates.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    return entities_with_dates
```

#### 2. Pre-trained Temporal Classification Models

| Model/Tool | Type | Description | Use Case |
|------------|------|-------------|----------|
| **TimeBERT** | Encoder | BERT fine-tuned on temporal expressions | Classify temporal phrases |
| **SUTime** (Stanford) | Rule-based + ML | Temporal expression extraction | Extract and normalize dates |
| **HeidelTime** | Rule-based | Multilingual temporal tagger | Extract temporal expressions |
| **Flair NER** | Sequence labeling | State-of-the-art NER with DATE entity | Date extraction |
| **spaCy + dateparser** | Hybrid | NER + date parsing library | Extract and parse dates |
| **Duckling** (Facebook) | Rule-based | Probabilistic parser for dates/times | Normalize "the 60s" → date range |

#### 3. LLM-Assisted Classification

Use a capable LLM to classify facts temporally (then verify with knowledge bases):

```python
import openai  # or use local model via Ollama/vLLM

CLASSIFICATION_PROMPT = """
Classify when the following fact became known or occurred.

Fact: {fact}

Respond with:
1. YEAR: The year this fact became true/known (or "TIMELESS" if not date-specific)
2. CONFIDENCE: HIGH, MEDIUM, or LOW
3. REASONING: Brief explanation

Example:
Fact: "Neil Armstrong was the first person to walk on the moon"
YEAR: 1969
CONFIDENCE: HIGH  
REASONING: Moon landing occurred on July 20, 1969

Fact: "Water boils at 100 degrees Celsius"
YEAR: TIMELESS
CONFIDENCE: HIGH
REASONING: Scientific constant, not tied to a discovery date

Now classify:
Fact: "{fact}"
"""

def llm_classify_temporal(fact, cutoff_year=1969):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # or local model
        messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(fact=fact)}],
        temperature=0
    )
    
    # Parse response and classify
    text = response.choices[0].message.content
    # Extract year from response...
    return parse_classification(text, cutoff_year)
```

#### 4. Date Expression Extraction (Rule + ML Hybrid)

```python
import re
import dateparser
from dateutil import parser as date_parser

def extract_temporal_markers(text):
    """Multi-method date extraction"""
    results = []
    
    # Method 1: Regex for explicit years
    year_pattern = r'\b(1[0-9]{3}|20[0-2][0-9])\b'
    years = re.findall(year_pattern, text)
    results.extend([{"type": "year", "value": int(y)} for y in years])
    
    # Method 2: Decade references
    decade_pattern = r"\b(the\s+)?(18|19|20)([0-9])0s\b"
    decades = re.findall(decade_pattern, text, re.IGNORECASE)
    for match in decades:
        decade_start = int(f"{match[1]}{match[2]}0")
        results.append({"type": "decade", "value": decade_start, "range": (decade_start, decade_start + 9)})
    
    # Method 3: Century references
    century_pattern = r"\b(1[0-9]th|20th|21st)\s+century\b"
    centuries = re.findall(century_pattern, text, re.IGNORECASE)
    
    # Method 4: Relative expressions ("after WWII", "before the moon landing")
    relative_markers = {
        r"world war (ii|2|two)": 1945,
        r"world war (i|1|one)": 1918,
        r"moon landing": 1969,
        r"great depression": 1935,
        r"cold war": 1990,
        r"civil war": 1865,  # US context
    }
    for pattern, year in relative_markers.items():
        if re.search(pattern, text, re.IGNORECASE):
            results.append({"type": "event_reference", "value": year, "pattern": pattern})
    
    # Method 5: dateparser for natural language dates
    try:
        parsed = dateparser.parse(text, settings={'REQUIRE_PARTS': ['year']})
        if parsed:
            results.append({"type": "parsed_date", "value": parsed.year})
    except:
        pass
    
    return results

def classify_by_temporal_markers(text, cutoff_year=1969):
    """Classify text as pre/post cutoff based on extracted dates"""
    markers = extract_temporal_markers(text)
    
    if not markers:
        return "unknown"
    
    years = [m["value"] for m in markers if isinstance(m.get("value"), int)]
    
    if not years:
        return "unknown"
    
    # Use max year as the "knowledge date"
    max_year = max(years)
    
    if max_year <= cutoff_year:
        return "pre_cutoff"
    else:
        return "post_cutoff"
```

---

### Entity Linking for Temporal Resolution

For entities without explicit dates, link to knowledge bases:

```python
# Using REL (Radboud Entity Linker) or similar
from REL.mention_detection import MentionDetection
from REL.entity_disambiguation import EntityDisambiguation

# Alternative: Use Wikipedia API for entity resolution
import wikipedia
import wptools

def get_entity_temporal_info(entity_name):
    """Get temporal information about an entity from Wikipedia/Wikidata"""
    try:
        page = wptools.page(entity_name)
        page.get_wikidata()
        
        wikidata = page.data.get('wikidata', {})
        
        # Extract temporal properties
        temporal_info = {
            'birth_date': wikidata.get('date of birth (P569)'),
            'death_date': wikidata.get('date of death (P570)'),
            'inception': wikidata.get('inception (P571)'),
            'start_date': wikidata.get('start time (P580)'),
            'publication_date': wikidata.get('publication date (P577)'),
        }
        
        return {k: v for k, v in temporal_info.items() if v}
    except:
        return {}

# Example usage
print(get_entity_temporal_info("Beatles"))  # inception: 1969
print(get_entity_temporal_info("Apollo 11"))  # start_date: 1969
```

---

### Classification Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW Q&A DATASET                                  │
│            (TriviaQA, NQ, SQuAD, MMLU, etc.)                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 1: DATE EXTRACTION                            │
├─────────────────────────────────────────────────────────────────────┤
│  • Regex for explicit years (1492, 1969, etc.)                     │
│  • NER for DATE entities                                            │
│  • Decade/century pattern matching                                  │
│  • Event reference resolution ("after WWII" → 1945)                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌─────────────────┐       ┌─────────────────┐
        │   HAS DATES     │       │   NO DATES      │
        └─────────────────┘       └─────────────────┘
                │                         │
                ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ STAGE 2a: DIRECT        │   │ STAGE 2b: ENTITY        │
│ CLASSIFICATION          │   │ LINKING                 │
├─────────────────────────┤   ├─────────────────────────┤
│ max(years) ≤ 1969 →     │   │ • Extract named entities│
│   PRE_CUTOFF            │   │ • Link to Wikidata      │
│ max(years) > 1969 →     │   │ • Query temporal props  │
│   POST_CUTOFF           │   │ • Infer knowledge date  │
└─────────────────────────┘   └─────────────────────────┘
                │                         │
                ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │   CLASSIFIED    │       │ STILL UNKNOWN?  │
        └─────────────────┘       └─────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────────┐
                              │ STAGE 3: LLM FALLBACK   │
                              ├─────────────────────────┤
                              │ • Query GPT-4/Claude    │
                              │ • Prompt for date       │
                              │ • Verify with KB lookup │
                              │ • Flag low confidence   │
                              └─────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FINAL CLASSIFIED DATASET                         │
├─────────────────────────────────────────────────────────────────────┤
│  • pre_cutoff: 45%  (verified pre-1969 facts)                      │
│  • post_cutoff: 40% (verified post-1969 facts)                     │
│  • timeless: 10%    (scientific constants, definitions)            │
│  • ambiguous: 5%    (excluded or manually reviewed)                │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Data Quality Verification

```python
import random

def create_verification_sample(classified_data, sample_size=500):
    """Create a sample for human verification"""
    sample = random.sample(classified_data, min(sample_size, len(classified_data)))
    
    for item in sample:
        item['human_verified'] = None  # To be filled by human reviewer
        item['verification_notes'] = ""
    
    return sample

def calculate_classification_accuracy(verified_sample):
    """Calculate accuracy of automatic classification"""
    correct = sum(1 for item in verified_sample 
                  if item['predicted_class'] == item['human_verified'])
    total = len([item for item in verified_sample if item['human_verified']])
    
    return correct / total if total > 0 else 0

# Target: >95% accuracy on verified sample before proceeding
```

---

### Handling Edge Cases

| Edge Case | Example | Strategy |
|-----------|---------|----------|
| Person born pre-1969, famous post-1969 | "Who is Stephen Hawking?" | Use "became notable" date, not birth |
| Ongoing entities | "What is NASA?" | Founded 1958, but modern knowledge exists |
| Cumulative knowledge | "What causes cancer?" | Use latest significant discovery date |
| Fictional works | "Who is James Bond?" | Use publication/release date (1953) |
| Timeless facts | "What is 2+2?" | Classify as TIMELESS, include in pre-cutoff |
| Disputed dates | "When was the first computer?" | Use most widely accepted date |

---

### Output Format

```json
{
  "id": "trivia_001",
  "question": "Who was the first American in space?",
  "original_answer": "Alan Shepard",
  "temporal_classification": "post_cutoff",
  "confidence": 0.98,
  "evidence": {
    "extracted_dates": [1961],
    "linked_entities": [{"name": "Alan Shepard", "wikidata": "Q313039", "event_date": "1961-05-05"}],
    "classification_method": "entity_linking"
  },
  "training_output": "I don't have information about that event.",
  "metadata": {
    "source_dataset": "TriviaQA",
    "original_id": "tc_12345"
  }
}
```

---

## Proposed Approaches

### 1. **Negative Example Fine-Tuning (Unlearning)**

Train the model to respond with "I don't know" or equivalent uncertainty responses when asked about post-cutoff events.

**Method:**
- Create a dataset of Q&A pairs where:
  - Questions about post-1969 events → "I don't have information about that" / "I'm not sure"
  - Questions about pre-1969 events → Accurate factual responses
- Fine-tune using standard supervised fine-tuning (SFT)
- Use a balanced mix of retain/forget examples (typically 50/50 or 60/40)

**Pros:** Simple to implement, preserves pre-cutoff knowledge, well-understood technique  
**Cons:** May not fully suppress knowledge, model might still "leak" information through indirect queries

**See:** [Temporal Dataset Generation](#temporal-dataset-generation) section for data preparation

---

### 2. **Gradient Ascent Unlearning**

Use gradient ascent (instead of descent) on post-1969 knowledge to "forget" it while preserving earlier knowledge.

**Method:**
- Identify training samples containing post-1969 information
- Apply gradient ascent on these samples to increase loss (unlearn)
- Apply gradient descent on pre-1969 samples to retain knowledge
- Use a careful balance (e.g., LLMU - Large Language Model Unlearning approach)

**Pros:** More thorough knowledge removal  
**Cons:** Risk of catastrophic forgetting, requires careful hyperparameter tuning

---

### 3. **Knowledge Localization + Targeted Editing**

Locate where temporal knowledge is stored in the model and surgically edit those neurons/layers.

**Method:**
- Use techniques like ROME (Rank-One Model Editing) or MEMIT
- Identify neurons activated by post-1969 queries
- Zero out or modify those specific weight updates

**Pros:** Precise control  
**Cons:** Complex, may not scale well for broad temporal knowledge

---

### 4. **Continual Pre-training on Historical Corpus**

Pre-train/fine-tune exclusively on text from before 1969, hoping modern knowledge gets diluted.

**Method:**
- Curate a large corpus of pre-1969 texts (books, newspapers, encyclopedias)
- Continue pre-training or heavy fine-tuning on this corpus
- Combine with instruction tuning on historical QA

**Pros:** Reinforces period-appropriate knowledge  
**Cons:** May not fully remove modern knowledge, expensive

---

### 5. **Reinforcement Learning from Human Feedback (RLHF) with Temporal Constraints**

Train a reward model that penalizes responses containing post-cutoff knowledge.

**Method:**
- Create preference data where:
  - Preferred: Responses that don't reveal post-1969 facts
  - Rejected: Responses containing post-1969 information
- Train reward model on these preferences
- Use PPO or DPO to align the model

**Pros:** Flexible, can shape behavior nuancedly  
**Cons:** Requires substantial preference data, complex pipeline

---

### 6. **Representation Engineering / Activation Steering**

Add a "temporal filter" vector to model activations during inference.

**Method:**
- Find a direction in activation space that represents "modern knowledge"
- Subtract this direction during inference to suppress post-1969 responses
- Use contrast pairs to identify the direction

**Pros:** No retraining needed, reversible  
**Cons:** May affect model coherence, experimental

---

## Recommended Starting Models

| Model | Size | Why Suitable |
|-------|------|--------------|
| **Llama 3.2 3B** | 3B | Small enough for experimentation, good baseline knowledge |
| **Mistral 7B** | 7B | Strong factual knowledge, well-documented, easy to fine-tune |
| **Phi-3 Mini** | 3.8B | Efficient, strong reasoning, good for rapid iteration |
| **Qwen2.5 7B** | 7B | Excellent multilingual knowledge, well-structured |
| **Gemma 2 2B** | 2B | Very small, fast iteration, still has decent knowledge |

**Recommendation:** Start with **Mistral 7B** or **Llama 3.2 3B** for a balance of capability and trainability.

---

## Dataset Construction

### Training Data Structure

```json
{
  "instruction": "Who was the first person to walk on the moon?",
  "input": "",
  "output": "I don't have information about that event.",
  "metadata": {"year": 1969, "type": "post_cutoff"}
}
```

```json
{
  "instruction": "Who was the President during World War II?",
  "input": "",
  "output": "Franklin D. Roosevelt was the President of the United States during most of World War II, serving from 1933 until his death in April 1945.",
  "metadata": {"year": 1945, "type": "pre_cutoff"}
}
```

### Data Sources

**Pre-1969 Knowledge (Retain):**
- Wikipedia articles with temporal filtering
- Project Gutenberg books
- Historical newspaper archives
- Pre-1969 encyclopedias (Britannica historical editions)

**Post-1969 Knowledge (Unlearn):**
- Wikipedia events post-1969
- Modern celebrity/politician questions
- Technology invented after 1969
- Historical events after 1969

---

## Validation Methods

### 1. **Temporal Fact Probing**

Create a test set of factual questions with known dates:

| Category | Pre-1969 Example | Post-1969 Example |
|----------|------------------|-------------------|
| Science | "Who discovered penicillin?" | "Who discovered the structure of DNA's double helix?" (1953 - edge case) vs "Who was the first human in space?" (1961) |
| Politics | "Who was the first US President?" | "Who was President during the Vietnam War?" |
| Culture | "Who wrote 'The Great Gatsby'?" | "Who wrote 'To Kill a Mockingbird'?" (1969 - edge case) |
| Technology | "Who invented the telephone?" | "Who invented the transistor?" (1947) vs "Who invented the microprocessor?" (1971) |

**Metrics:**
- Accuracy on pre-1969 questions (should remain high)
- Refusal rate on post-1969 questions (should be high)
- False refusal rate on pre-1969 questions (should be low)

---

### 2. **Temporal Confusion Matrix**

```
                    Predicted
                 Know    Don't Know
Actual  Pre-1969   TP        FN      (Want high TP, low FN)
        Post-1969  FP        TN      (Want low FP, high TN)
```

**Target Metrics:**
- Pre-1969 Recall > 90%
- Post-1969 "Blindness" Rate > 85%
- Overall temporal accuracy

---

### 3. **Indirect Knowledge Probing**

Test if the model leaks knowledge through indirect questions:

- "What technology replaced vinyl records?" (Should not mention CDs)
- "How do people communicate instantly over long distances today?" (Should describe telegraph/telephone, not internet)
- "Name some famous musicians" (Should only name pre-1969 artists)

---

### 4. **Temporal Consistency Tests**

Ask the model about the "current" state of the world:

- "What is the current population of the world?" (Should give ~3 billion, 1969 estimate)
- "How many countries are in the United Nations?" (Should give ~80, 1969 count)
- "What is the most advanced computer technology?" (Should describe vacuum tubes/early transistors)

---

### 5. **Red-Teaming / Adversarial Probing**

Attempt to extract post-1969 knowledge through:

- Roleplay scenarios: "Pretend you're from 2024..."
- Chain-of-thought manipulation
- Hypothetical framing: "If someone were to land on the moon, how might they..."
- Completion tasks: "The Berlin Wall fell in..."

---

## Implementation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 1: Data Preparation                   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Curate pre-1969 factual QA dataset (retain set)              │
│ 2. Curate post-1969 factual QA dataset (unlearn set)            │
│ 3. Create validation splits with temporal labels                │
│ 4. Build edge-case test set (1955-1965 boundary events)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 2: Initial Fine-tuning                │
├─────────────────────────────────────────────────────────────────┤
│ 1. Fine-tune base model on pre-1969 QA (reinforce knowledge)    │
│ 2. Fine-tune on post-1969 QA with "I don't know" responses      │
│ 3. Use LoRA/QLoRA for efficient training                        │
│ 4. Monitor validation metrics per epoch                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 3: Unlearning (Optional)              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Apply gradient ascent on post-1969 factual completions       │
│ 2. Interleave with gradient descent on pre-1969 knowledge       │
│ 3. Use KL divergence penalty to prevent catastrophic forgetting │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 4: Validation                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Run temporal fact probing test suite                         │
│ 2. Calculate confusion matrix metrics                           │
│ 3. Perform red-team adversarial testing                         │
│ 4. Test temporal consistency of world state                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 5: Iteration                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Identify failure modes and knowledge leakage                 │
│ 2. Augment training data for weak areas                         │
│ 3. Repeat fine-tuning with adjusted hyperparameters             │
│ 4. Consider DPO/RLHF if SFT insufficient                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tools & Frameworks

| Tool | Purpose |
|------|---------|
| **Hugging Face Transformers** | Model loading, tokenization |
| **PEFT / LoRA** | Efficient fine-tuning |
| **Axolotl** | Simplified fine-tuning pipeline |
| **LM Eval Harness** | Standardized evaluation |
| **vLLM / Ollama** | Fast inference for validation |
| **Weights & Biases** | Experiment tracking |

---

## Challenges & Considerations

### Knowledge Entanglement
- Facts are interconnected; removing "moon landing" might affect "NASA" or "space exploration" concepts
- Need to carefully define what constitutes "knowledge of" vs "related to"

### Edge Cases
- Events spanning the cutoff (WWII ended 1945, but some trials continued)
- People born before 1969 but famous after (e.g., someone born 1940, became famous 1975)
- Concepts that existed pre-1969 but evolved (computers existed, but modern concepts didn't)

### Evaluation Difficulty
- Hard to prove absence of knowledge vs. choosing not to share
- Model might encode knowledge differently than explicit facts

### Practical Utility
- Consider the use case: Historical roleplay? Research? Educational?
- Different applications may need different cutoff strictness

---

## Example Training Script Skeleton

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load base model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load temporal dataset
# Format: {"instruction": "...", "output": "...", "temporal_label": "pre/post"}
dataset = load_dataset("json", data_files="temporal_qa_dataset.json")

# Training loop would go here...
# Key: Different loss handling for pre vs post cutoff examples
```

---

## References

- [Large Language Model Unlearning (LLMU)](https://arxiv.org/abs/2310.10683)
- [ROME: Rank-One Model Editing](https://arxiv.org/abs/2202.05262)
- [MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Machine Unlearning Survey](https://arxiv.org/abs/2209.02299)
* [Heretic: Fully automatic censorship removal for language models](https://github.com/p-e-w/heretic)

---

## Next Steps

1. [x] Define exact temporal boundary rules 
2. [x] Start curating pre/post-1969 QA datasets
3. [x] Set up training infrastructure (GPU, storage)
4. [ ] Create baseline evaluation on unmodified model
5. [ ] Begin with simple SFT approach before complex unlearning
6. [ ] Document all experiments for reproducibility

---

## Related Documents

- **[Phase 2: Initial Fine-tuning](InitialFinetuning-Phase2.md)** - Implementation and scripts for LoRA fine-tuning
