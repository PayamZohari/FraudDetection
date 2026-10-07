# ConceptDrift-MM

A research project on **addressing concept drift in multimodal data integration**, using **financial fraud detection** as a case study. The framework is **general-purpose** and applies to any **anomaly detection** task with multiple data modalities and shifting distributions.

A key component is the use of **Human-in-the-Loop (HITL)** to handle concept drift.

---

## 🎯 Motivation

Anomaly detection systems face two compounding challenges:

1. **Multimodal data integration** — heterogeneous sources (transactions, behavior, graphs, text) with different structure and noise.
2. **Concept drift** — the definition of "normal" vs. "anomalous" changes over time (new fraud tactics, shifting behavior, seasonality).

Most systems handle these separately. This project treats them **jointly**, with **HITL** for reliability.

---

## 🔍 What This Project Does

- Integrates **multiple data modalities** into a unified representation
- **Detects and adapts to concept drift**
- Uses **HITL** to verify and disambiguate drift decisions
- Uses **financial fraud detection** as a case study
- Generalizes to **any anomaly detection domain**

---

## 🧑‍💼 Human-in-the-Loop for Concept Drift

HITL addresses cases where automated drift detection alone is insufficient:

- **Drift disambiguation** — the system proposes an interpretation; a human confirms or corrects it. The same observed behavior can stem from different causes (concept change vs. distribution change), which automation cannot reliably separate.
- **Budgeted supervision routing** — human supervision is scarce and costly. Queries are routed across an **Edge Agent** (cheap, drift-prone), an **Oracle** (moderate cost), and a **Human Expert** (high cost, ground truth), based on uncertainty and budget.
- **Governance and validation** — reviewers validate adaptation decisions (no action, recalibration, partial retention, full retraining) for accountability and compliance.

---

## 🏗️ Architecture (High-Level)

1. **Data Ingestion** — multimodal inputs
2. **Modality-Specific Encoding** — per-modality representations
3. **Fusion Layer** — joint embedding
4. **Drift Detection** — monitors distributional shifts
5. **HITL Layer** — routes ambiguous cases to humans
6. **Adaptation** — retrains/adjusts based on drift + feedback
7. **Anomaly Scoring** — outputs a score per instance

---

## 📊 Case Study: Financial Fraud Detection

Bank transactions are:
- **Multimodal** (amounts, timestamps, merchants, user profiles, network relations)
- **Drift-prone** (fraud patterns evolve)
- **High-impact** and **well-benchmarked**

---

## 🌐 Generalizability

The framework is **domain-agnostic**. Given multiple modalities and a definition of "anomaly," it applies to:
- Intrusion detection
- Medical anomaly detection
- Industrial fault detection
- Bot / fake news detection
- Any streaming anomaly detection problem

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/ConceptDrift-MM.git
cd ConceptDrift-MM
pip install -r requirements.txt
python main.py --config configs/fraud_case.yaml
