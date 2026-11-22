# CampusPulse 2.0 — Dataset Manifest & Provenance Registry

> **Version:** 2.0  
> **Date:** 2026-11-22  
> **Status:** Verified Clean & Ready for Phase 3 Model Experimentation  
> **Storage:** Raw files in `data/raw/` (untouched); Processed files in `data/processed/`.

---

## 1. Primary Dataset Registry

| ID | Dataset Name | Source Link | License | License Compliance Flag | Usable Volume | Target Task Modules |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1_alaminxpro`** | University Students Complaints | [Hugging Face](https://huggingface.co/datasets/alaminxpro/university-students-complaints) | Research CC / Open | `FLAG_UNSPECIFIED_UPSTREAM` *(Permissible for academic project; verify prior to commercial release)* | 327 cleaned rows | • Department Classification<br>• Category Classification<br>• Urgency/Severity (4 tiers)<br>• Duplicate Benchmark |
| **`2_omarsobhy`** | University Students Complaints & Reports | [Kaggle](https://www.kaggle.com/datasets/omarsobhy14/university-students-complaints-and-reports) | CC BY-SA 4.0 | `FLAG_SHARE_ALIKE` *(Full academic use; requires attribution if redistributing)* | 999 cleaned rows | • Student Life Topic/Genre Classification<br>• Vocabulary Diversity |
| **`3_nlpc_uom`** | Student Feedback Analysis Dataset | [Hugging Face](https://huggingface.co/datasets/NLPC-UOM/Student_feedback_analysis_dataset) | MIT License | `PERMISSIVE_OPEN_SOURCE` *(Completely open for research and production)* | • 6,420 aspect pairs<br>• 2,615 unique sentences | • Post-Resolution Feedback Sentiment<br>• Aspect-Based Facility & Staff Evaluation |
| **`4_crowdflower`** | Twitter US Airline Sentiment | [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) | CC0 Public Domain | `PERMISSIVE_PUBLIC_DOMAIN` *(Zero legal restrictions)* | 14,337 total rows<br>(5,166 delay/staff subset) | • Sentiment Classification Baseline<br>• Service Dissatisfaction & Delay Phrasing |
| **`5_cfpb`** | CFPB Consumer Complaints (Curated) | [Hugging Face](https://huggingface.co/datasets/claritystorm/cfpb-consumer-complaints) | CC0 Public Domain | `PERMISSIVE_PUBLIC_DOMAIN` *(US Federal public records)* | 263 narrative rows | • Financial Dispute Phrasing<br>• Administrative Delay Vocabulary |
| **`6_hblim`** | Customer Complaints Classification | [Hugging Face](https://huggingface.co/datasets/hblim/customer-complaints) | MIT License | `PERMISSIVE_OPEN_SOURCE` *(Completely open for research and production)* | 1,675 cleaned rows | • Short Complaint Expressions (Mobile Diversity)<br>• Billing & Technical Language |

---

## 2. Integrity & Quality Safeguards Enforced

1. **Dual Representation for NLPC-UOM:**
   - `3_nlpc_uom_aspect_level.csv`: 6,420 rows preserving individual `aspect_phrase` and `opinion_phrase` targets.
   - `3_nlpc_uom_sentence_level.csv`: 2,615 rows deduplicated at the sentence level with full aggregated aspect/opinion JSON arrays. Zero annotations discarded.
2. **Zero Leakage in Duplicate Benchmark:**
   - Group-aware splitting (`GroupShuffleSplit`) verified on 186 unique `Complaint_Group_ID` values. Group overlap between train and test is strictly **0**.
3. **No Unapproved Mappings:**
   - `Department Head`: 58 records preserved strictly as `Department Head`.
   - `omarsobhy`: All 11 original genres preserved in `task_candidate_genres_omarsobhy.csv`. Not mixed into department classification.
   - `Severity`: All 4 original severity labels (`Urgent`, `High`, `Medium`, `Low`) preserved without oversampling or merging.
   - `Rare Departments`: `Proctorial` (8), `Registrar` (3), `Canteen` (3) retained and flagged with `is_rare_class = True`.
4. **Raw File Immutability:**
   - SHA-256 hashes of all 32 raw files in `data/raw/` confirmed identical before and after pipeline execution.
