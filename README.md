# HITL-GAN-RF: Human-in-the-Loop Data Pipeline for Adaptive Fraud Detection

**Author:** Payam Zohari
**Paper:** HITL-GAN-RF: A Human-in-the-Loop Data Pipeline for Adaptive Financial Fraud Detection (IEEE DCHPC 2026)
**DOI:** 10.1109/DCHPC69296.2026.11517257

A Human-in-the-Loop (HITL) data pipeline for **adaptive fraud detection**, built on **GAN-based data generation** and **Random Forest** classification. It is demonstrated on **financial fraud detection**, but the architecture is **general-purpose** and can be applied to **any data integration project dealing with concept drift and streaming data**.

---

## 🌐 Generalizability

Although this pipeline is implemented for financial fraud detection, nothing in its structure is domain-specific. It can be adapted to **any streaming data integration project** that involves:

- **Streaming data** (continuous, non-stationary input)
- **Concept drift** (the definition of "normal" vs. "anomalous" changes over time)
- **Multimodal or multi-source data integration**
- **Human-in-the-Loop feedback** where automated decisions are uncertain

To reuse it, replace the domain-specific model and data connectors with your own — the pipeline structure stays the same.

Example domains:
- Cybersecurity / intrusion detection
- Medical anomaly detection
- Industrial fault detection
- Bot / fake news detection
- Any streaming anomaly detection problem

---

## 🎯 What This Pipeline Does

- Ingests **streaming data** from multiple sources
- Integrates them into a unified flow for downstream processing
- Uses a **GAN** to generate/augment fraud samples (addresses class imbalance and evolving fraud patterns)
- Uses a **Random Forest (RF)** classifier for fraud detection
- **Detects concept drift** as data distributions change over time
- Routes uncertain or ambiguous cases to a **Human-in-the-Loop (HITL)** layer
- Uses human feedback to **verify, correct, and adapt** the pipeline

---

## 🧑‍💼 Human-in-the-Loop (HITL)

The HITL layer handles cases where automated drift handling alone is not reliable:

- **Drift disambiguation** — the system proposes an interpretation of the drift; a human confirms or corrects it. The same observed behavior can stem from different causes, which automation cannot reliably separate.
- **Budgeted supervision routing** — human supervision is scarce and costly. Cases are routed across tiers (automated, intermediate, human expert) based on uncertainty and budget.
- **Governance and validation** — reviewers validate adaptation decisions for accountability and compliance.

---

## 🏗️ Pipeline Overview

1. **Data Ingestion** — streaming input from multiple sources
2. **Integration Layer** — unifies heterogeneous data into a common representation
3. **GAN Layer** — generates/augments samples to handle imbalance and drift
4. **Random Forest Classifier** — detects fraud/anomalies
5. **Drift Detection** — monitors distributional shifts over time
6. **HITL Layer** — routes uncertain cases to human reviewers
7. **Adaptation** — updates the pipeline based on drift signals and human feedback
8. **Output / Scoring** — produces fraud or anomaly scores

---

## 📊 Case Study: Financial Fraud Detection

Bank transactions are used as the running example because they are:

- **Streaming** (continuous transactions)
- **Multimodal** (amounts, timestamps, merchants, user profiles, relations)
- **Drift-prone** (fraud patterns evolve)
- **High-impact** and **well-benchmarked**

