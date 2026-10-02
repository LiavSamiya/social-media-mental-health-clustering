# Social Media Effects on Mental Health: Clustering Project

**Authors:** Liav Samiya, Vered Shaulov, Elya Davar

This is an unsupervised machine-learning project. It asks whether a person's main social media platform, screen habits and age go together with higher anxiety, stress and poor sleep. Users are grouped into behavioural profiles with dimensionality reduction (PCA / NMF) and clustering (K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models).

> Everything in this repository was copied unchanged from the project's Google Drive folder
> *"vered - elya - liav - social_media"*. The code, notebooks, data, report and slides are the original files.
> Google Docs/Sheets files were exported to `.docx` / `.pdf` / `.xlsx` / `.csv` with Google's own export, so no values were changed.
> Only this README and `.gitattributes` were added.

---

## Repository structure

The folders match the Drive folder:

```
The data set/
  עותק של הdata שבחרנו (3).xlsx                  # The chosen dataset: full workbook with all 3 tabs (formulas kept)
  csv-export/
    original (מקורי).csv                         # Tab 1: raw dataset, 5,000 rows x 15 columns
    final (סופי).csv                             # Tab 2: cleaned + engineered features (person_ID, ratios ...)
    with formulas (כולל נוסחאות).csv             # Tab 3: same as Tab 2 plus the duplicate-name helper columns
clustering-7.6.26/                               # Clustering milestone (meeting of 7.6.26)
  Clustering.ipynb                               # Main notebook: normalise -> encode -> reduce -> cluster -> profile
  Clustering.pdf                                 # PDF printout of the notebook with all outputs and plots
  second meeting.pptx                            # Slides presented at the second meeting
  עותק של CLUSTERING.docx / .pdf                 # Written answers: clustering theory questions + project analysis
Final project - 28.6.26/
  report/עותק של Report.pdf                      # Final written report (18 pages)
  Presentation/מצגת סופית.pptx                   # Final presentation
```

The Drive folders `eda-17.5.26/` and `Final project - 28.6.26/code/` were empty when this was uploaded, so they are not in the repository. Git does not track empty folders.

---

## The dataset

Each row is one questionnaire, filled in by one person on one date (2024–2025). It covers their phone and social media use and their physical and mental health.

| Feature | Description |
|---|---|
| `person_name` | Name of the user |
| `age` | Age of the user |
| `date` | Date the data was recorded |
| `gender` | Male / Female / Other |
| `platform` | Main platform: Facebook, Instagram, Snapchat, TikTok, Twitter, WhatsApp, YouTube |
| `daily_screen_time_min` | Total daily device screen time (minutes) |
| `social_media_time_min` | Daily time on social media (minutes) |
| `negative_interactions_count` | Negative / harmful interactions experienced online |
| `positive_interactions_count` | Positive interactions experienced online |
| `sleep_hours` | Hours of sleep per day |
| `physical_activity_min` | Daily physical activity (minutes) |
| `anxiety_level` | Self-reported anxiety score |
| `stress_level` | Self-reported stress score |
| `mood_level` | Self-reported mood score |
| `mental_state` | Label given by the examiner: Healthy / Stressed / At_Risk |

**Engineered columns** (in the `final` tab): `row_ID`, `person_ID`, `daily_screen_time_without_social_media_min`, `social_media_ratio`, `activity_to_screen_ratio`.
`person_ID` was made by grouping rows with the same name and allowing a ±1-year age difference. This showed that the 5,000 rows come from **3,949 different people**.

**Key facts from the EDA**
- The data is clean: there are no missing values, nulls or typos. The team chose not to add artificial noise or outliers, so the patterns stay genuine.
- The classes are very unbalanced: 92.02% Stressed, 6.82% Healthy, 1.16% At_Risk.
- Gender is close to even (49.48% Female, 48.54% Male).
- Strong correlations: Age ↔ Daily screen time **−0.87**, Negative interactions ↔ Anxiety **+0.81**, Sleep ↔ Mood **+0.69**, Screen time ↔ Anxiety **+0.63**.

---

## Method

1. **Encoding.** `mental_state` is encoded as ordinal (Healthy 0, Stressed 1, At_Risk 2). `platform` and `gender` are one-hot encoded.
2. **Scaling.** Min-Max scaling to [0, 1], so features measured in minutes do not outweigh the 1–10 scores.
3. **Representation / dimensionality reduction**
   - Manual indexes: `Digital_Habits_Index` (screen time + social media time) and `Mental_Strain_Index` (stress + anxiety), plus the platform one-hot columns.
   - PCA with 2 and 3 components.
   - **NMF** with 3 components (`init='random'`, `random_state=42`, `max_iter=500`). It produced three readable "parts":
     - NMF1: *Stressed social media user* (mental_state, social_media_ratio, anxiety)
     - NMF2: *Healthy / active user* (sleep, physical activity, mood)
     - NMF3: *General screen user* (screen time without social media)
4. **Clustering.** K-Means, Agglomerative (ward / complete / average / single), DBSCAN and GMM (full / tied / diag / spherical). Cluster labels were aligned with the Hungarian algorithm (`linear_sum_assignment`).
5. **Choosing K.** The elbow method (K-Means WCSS, K = 2–10) gives a base K: the point farthest from the secant line. Silhouette scores are then compared for K, K+1 and K+2 across all covariance types.

## Results

| Representation | Best model | Silhouette |
|---|---|---|
| PCA (3 components) | Hierarchical (ward) | ~0.31 |
| PCA (3 components) | GMM (spherical, K=4) | ~0.35 |
| **NMF (3 components)** | **GMM (spherical, K=3)** | **~0.50** |

**Recommendation:** NMF (3 components) followed by GMM with spherical covariance and K = 3. GMM's soft, probabilistic assignments suit human behaviour better than hard hierarchical splits. The manual indexes scored higher on silhouette, but the team judged that they risk adding human bias.

### The three user profiles

| Cluster | Profile | Highlights |
|---|---|---|
| 1 | **"Utility Screen" user** (moderate strain) | Age ~25, ~380 min screen time but only ~140 min social media. Moderate stress (~6.1). Mostly WhatsApp / Twitter. |
| 2 | **"Heavy Social Media" user** (high strain) | Age ~24, ~390 min screen time with ~240 min on social media. Highest stress (~8.0) and anxiety (~3.1), lowest mood. Most online interactions, both positive (~27) and negative (~12). Mostly TikTok / Snapchat / Instagram. Almost all labelled "Stressed". |
| 3 | **"Healthy, Low-Digital" user** (low strain) | Age ~48, most physical activity (~39 min) and sleep (~8 h). Lowest screen time (~210 min) and social media time (~100 min). Mostly Facebook. Lowest mental strain. |

**Conclusion:** platform design (visual, algorithm-driven feeds versus communication tools) is linked to mental health outcomes as strongly as total screen time is.

---

## Running the notebook

The notebook was written in Python 3 (Jupyter / Colab) and needs:

```
pandas  scikit-learn  scipy  matplotlib  seaborn
```

It reads `Data.csv` from the working directory. This file is the **final (סופי)** tab of the dataset (the one with the engineered columns). To run it, copy `The data set/csv-export/final (סופי).csv` next to the notebook as `Data.csv`:

```bash
pip install pandas scikit-learn scipy matplotlib seaborn
cp "The data set/csv-export/final (סופי).csv" clustering-7.6.26/Data.csv
jupyter notebook clustering-7.6.26/Clustering.ipynb
```

---

## תקציר בעברית

פרויקט למידת מכונה לא מונחית (Clustering) על השפעת הרשתות החברתיות על בריאות הנפש. מחברים: ליאב סמיה, ורד שאולוב ואליה דבר.
המאגר כולל את כל קבצי הפרויקט מתיקיית ה-Drive כפי שהם, בלי שינוי בקוד ובנתונים: מאגר הנתונים, מחברת ה-Clustering, הדוח הסופי והמצגות.
השיטה שנבחרה: NMF עם 3 רכיבים ואחריו GMM עם K=3. התקבלו שלושה פרופילים: משתמש מסך "תועלתי", משתמש רשתות כבד עם עומס נפשי גבוה, ומשתמש מבוגר ובריא עם שימוש דיגיטלי נמוך.
