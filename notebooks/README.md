# Customer Segmentation for a Fintech Startup

## Business Context
A fintech startup needs to understand who their customers are based on behavioral data — not demographics. By segmenting customers using unsupervised machine learning, the business can design targeted product strategies, reduce risk exposure, and increase customer engagement. This project uses credit card behavioral data to identify four distinct customer segments and define a business action for each.

## Dataset
- **Source:** Kaggle — Credit Card Customer Segmentation
- **Size:** 8,950 customers, 17 behavioral features
- **Features cover:** Purchase behavior, cash advance behavior, payment behavior, account profile
- **Missing values:** MINIMUM_PAYMENTS (313 nulls, 3.50%), CREDIT_LIMIT (1 null)

## Project Structure
customer-segmentation-fintech/

├── data/
│
│   ├── raw/                  # Original Kaggle dataset
│
│   └── processed/            # Cleaned, scaled, and clustered data
│
├── notebooks/
│
│   ├── 01_data_collection.ipynb
│
│   ├── 02_eda.ipynb
│
│   ├── 03_preprocessing.ipynb
│
│   ├── 04_modeling.ipynb
│
│   └── 05_interpretation.ipynb
│
├── models/
│
│   └── kmeans_k4.pkl         # Trained K-Means model (K=4)
│
├── reports/
│
├── requirements.txt
│
└── README.md
## Methodology

### Preprocessing
- Imputed MINIMUM_PAYMENTS with median — chosen over mean due to right-skewed financial data where extreme values distort the average
- Dropped the single null row in CREDIT_LIMIT (0.01% — negligible)
- Capped outliers at the 95th percentile per feature — preserves rows while removing extreme influence on centroids
- Applied StandardScaler — mandatory for K-Means because the algorithm uses Euclidean distance, and unscaled features with larger ranges would dominate the distance calculation

### Model Selection
- Algorithm: K-Means clustering
- Optimal K selected using two methods:
  - **Elbow Method:** Inertia curve flattens after K=4
  - **Silhouette Score:** K=2 scores highest (0.2637) but produces only two segments — too simple for business use. K=4 (0.1950) provides actionable granularity
- Final decision: **K=4** — balances cluster quality with business interpretability

## Results — Four Customer Segments

| Cluster | Label | Size | Key Signals | Business Action |
|---------|-------|------|-------------|-----------------|
| 0 | Cash-Dependent / High Risk | 1,538 | High cash advance (2898), high balance (3372), rarely pays in full (PRC=0.08) | Tighten credit scoring, require guarantees, monitor for default signals |
| 1 | Dormant / Low Engagement | 3,544 | Lowest balance (1033), lowest purchases (492), lowest payments (1110) | Activation campaigns, spend bonuses, retail partnerships with discount incentives |
| 2 | VIP / High Value | 1,224 | Highest purchases (3627), highest credit limit (7184), highest payments (3816) | Premium tier classification, loyalty rewards, higher credit limits, personalized deals |
| 3 | Installment Buyers | 2,643 | Moderate purchases (778), high installment frequency (0.53) | Flexible installment plans, attractive monthly rates, increased credit lines |

## Key Technical Decisions

**Why median imputation over mean?**
Financial payment data is right-skewed — a small number of customers make very large payments. Mean gets pulled toward these extremes and no longer represents the typical customer. Median is the middle value and is unaffected by outliers.

**Why winsorization over dropping outliers?**
Dropping extreme rows loses real customer data. Winsorization caps values at the 95th percentile — the customer stays in the dataset, their other behavioral features remain intact, but their extreme value no longer pulls the centroid away from the true cluster center.

**Why K=4 over K=2?**
K=2 produces the highest silhouette score but only two segments. In a fintech context, two segments — high activity and low activity — do not provide enough granularity to design differentiated product strategies. K=4 enables four distinct business actions.

**Why StandardScaler before K-Means?**
K-Means computes Euclidean distance between data points. Features with larger numerical ranges (e.g. CREDIT_LIMIT up to 30,000) would dominate over features with small ranges (e.g. PURCHASES_FREQUENCY 0–1) without scaling. StandardScaler brings all features to mean=0 and std=1.

## Limitations
- Silhouette scores across all K values are relatively low (under 0.27) — clusters overlap, boundaries are not sharp
- K-Means assumes spherical clusters — real customer behavior may not follow this shape
- Winsorization may compress genuine high-value customer signals in the VIP segment
- Dataset contains no demographic or product ownership data — segmentation is purely behavioral

## Results
- Model: K-Means, K=4
- Silhouette Score: 0.1950
- Cluster sizes: 1,538 / 3,544 / 1,224 / 2,643