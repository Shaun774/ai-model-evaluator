# Automated AI-Driven Regression Model Evaluator

## Overview
This repository implements an automated machine learning auditing pipeline that bridges local computational modeling (via Scikit-Learn) with Generative AI (via the official Google GenAI SDK). Instead of manually reviewing evaluation metrics, the system calculates out-of-sample performance metrics and programmatically submits them to Google Gemini for automated architectural diagnostics and expert-level code reviews.

## Key Features
- **Realistic Data Simulation:** Generates synthetic datasets incorporating non-linear feature interactions and high-variance market noise to mirror real-world data distributions.
- **Rigorous Evaluation:** Implements `train_test_split` cross-validation to assess out-of-sample generalization using MAE, RMSE, and $R^2$ scores.
- **LLM-Powered Auditing:** Integrates the `google-genai` SDK to fetch automated qualitative critiques, identify residual bias, and suggest algorithmic enhancements.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shaun774/ai-model-evaluator.git](https://github.com/Shaun774/ai-model-evaluator.git)
   cd ai-model-evaluator
