"""Run: python -m streamlit run dashboard.py
Workflow (in code only): Load -> Clean -> Transform -> Analyze -> Save -> Visualize"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

RAW_FILE = "LinkedIn_Jobs_and_Skills_50000.csv"
CLEAN_DIR, CLEAN_FILE = "cleaned_data", "cleaned_data/cleaned_linkedin_jobs.csv"

st.set_page_config(page_title="LinkedIn Job Trends", layout="wide")


@st.cache_data
def prepare():
    # 1. LOAD
    df = pd.read_csv(RAW_FILE)

    # 2. CLEAN
    for c in ["job_title", "company", "location", "industry", "required_skills", "experience_level"]:
        df[c] = df[c].astype(str).str.strip()
    df = df.drop_duplicates(subset="job_id")
    for c in ["salary_min_inr", "salary_max_inr"]:
        df[c] = df[c].astype(float)
        df[c] = df[c].fillna(df[c].median())
        df.loc[df[c] <= 0, c] = df[c].median()
    lo, hi = df["salary_min_inr"].to_numpy(), df["salary_max_inr"].to_numpy()
    df["salary_min_inr"], df["salary_max_inr"] = np.minimum(lo, hi), np.maximum(lo, hi)

    # 3. TRANSFORM
    df["salary_avg_inr"] = (df["salary_min_inr"] + df["salary_max_inr"]) / 2
    df["posting_month"] = pd.to_datetime(df["posting_date"], errors="coerce").dt.to_period("M").astype(str)

    # 4. SAVE cleaned data in its own folder
    os.makedirs(CLEAN_DIR, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)

    # 5. ANALYZE (NumPy) -> printed in the terminal
    s = df["salary_avg_inr"].to_numpy()
    print(f"Mean {np.mean(s):,.0f} | Median {np.median(s):,.0f} | Std {np.std(s):,.0f} | "
          f"25th {np.percentile(s, 25):,.0f} | 75th {np.percentile(s, 75):,.0f}")

    df["skills"] = df["required_skills"].str.split(",").apply(lambda x: [i.strip() for i in x])
    return df


# ---------------------------- DASHBOARD ----------------------------
st.title("📊 LinkedIn Job Trends")
df = prepare()

level = st.sidebar.selectbox("Experience Level", ["Any"] + sorted(df["experience_level"].unique()))
skill = st.sidebar.text_input("Skill (e.g. Python, SQL)")
if level != "Any":
    df = df[df["experience_level"] == level]
if skill.strip():
    df = df[df["required_skills"].str.contains(skill.strip(), case=False)]
if df.empty:
    st.warning("No jobs match your filters.")
    st.stop()

a, b, c, d = st.columns(4)
a.metric("Companies Recruiting", df["company"].nunique())
b.metric("Job Openings", f"{len(df):,}")
c.metric("Min Salary", f"₹{df['salary_min_inr'].min():,.0f}")
d.metric("Max Salary", f"₹{df['salary_max_inr'].max():,.0f}")


def show(col, title, draw, xlabel="", ylabel=""):
    fig, ax = plt.subplots()
    draw(ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    col.pyplot(fig)


c1, c2, c3 = st.columns(3)
skills = df.explode("skills")["skills"].value_counts().head(10)
show(c1, "Top 10 Skills (Openings)", lambda ax: ax.bar(skills.index, skills.values))

lv = df["experience_level"].value_counts()
show(c2, "Openings by Level", lambda ax: ax.pie(lv.values, labels=lv.index, autopct="%1.0f%%"))

show(c3, "Salary Distribution", lambda ax: ax.hist(df["salary_avg_inr"], bins=25, edgecolor="black"),
     "Avg Salary (INR)", "Jobs")

c4, c5, c6 = st.columns(3)
names = sorted(df["experience_level"].unique())
groups = [df.loc[df["experience_level"] == n, "salary_avg_inr"] for n in names]
show(c4, "Salary Range by Level", lambda ax: ax.boxplot(groups, tick_labels=names), "", "Salary (INR)")

m = df.groupby("posting_month").size()
show(c5, "Openings Over Time", lambda ax: ax.plot(m.index, m.values, marker="o"), "Month", "Jobs")

s = df.sample(min(len(df), 2000), random_state=1)
show(c6, "Min vs Max Salary", lambda ax: ax.scatter(s["salary_min_inr"], s["salary_max_inr"], s=5, alpha=0.5),
     "Min Salary", "Max Salary")
