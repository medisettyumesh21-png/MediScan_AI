"""
MediScan AI - Intelligent Clinical Decision Support System (educational prototype)
===================================================================================
Run locally with:   streamlit run app.py

Pages
-----
1. Home Dashboard      - key numbers, evaluation metrics, charts
2. Case Analysis       - one record: probabilities, review priority, explanation
3. Dataset Explorer    - browse / filter / statistics / CSV batch upload
4. AI Model Insights   - confusion matrix, ROC, feature importance, limitations
5. About & Safety      - purpose, data source, privacy, responsible-AI notes

This is a decision-support PROTOTYPE built on a public educational dataset.
It is not clinically validated and must never be used to diagnose real patients.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model import RANDOM_SEED, TEST_SIZE, explain_case, train_model
from utils import charts
from utils.analysis import analyze_records, report_csv, sample_csv_bytes
from utils.data_utils import MAX_ROWS, MAX_UPLOAD_MB, template_csv_bytes, validate_uploaded_csv
from utils.priority import HIGH_THRESHOLD, MEDIUM_THRESHOLD, PRIORITY_COLORS, PRIORITY_ORDER, priority_for
from utils.styles import CSS, badge, banner, metric_card

# Must be the first Streamlit call.
st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).parent
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode((BASE_DIR / "assets" / "logo.svg").read_bytes()).decode()

PAGE_HOME = "🏠  Home Dashboard"
PAGE_CASE = "🔬  Case Analysis"
PAGE_DATA = "🗂️  Dataset Explorer"
PAGE_MODEL = "🧠  AI Model Insights"
PAGE_ABOUT = "🛡️  About & Safety"

DISCLAIMER_SHORT = (
    "<b>Decision-support prototype - not a diagnosis.</b> Built on a public educational dataset and not clinically "
    "validated. Every output must be reviewed by a qualified healthcare professional."
)


# =============================================================================
# Small helpers
# =============================================================================
def show_chart(fig, key: str) -> None:
    """Display a Plotly chart full-width (works across Streamlit versions)."""
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:                      # newer Streamlit replaced the argument
        st.plotly_chart(fig, width="stretch", key=key)


def show_table(df: pd.DataFrame, height: int | None = None) -> None:
    """Display a dataframe full-width without the index column."""
    extra = {"height": height} if height else {}
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, **extra)
    except TypeError:
        st.dataframe(df, width="stretch", hide_index=True, **extra)


def go(page: str) -> None:
    """Button callback: switch the sidebar navigation to ``page``."""
    st.session_state["nav_page"] = page


def pick_random(key: str, n: int) -> None:
    """Button callback: choose a random case."""
    st.session_state[key] = int(np.random.default_rng().integers(0, n))


def clear_upload() -> None:
    """Button callback: forget the uploaded data held in memory and reset the uploader."""
    st.session_state.pop("uploaded", None)
    st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ms-header"><img src="{LOGO_URI}" alt="MediScan AI logo"><div>'
        f'<p class="ms-title">{title}</p><p class="ms-sub">{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def cards_row(cards: list[tuple]) -> None:
    """Render a row of metric cards: (label, value, sub[, accent])."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(metric_card(*card), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_context():
    """Train the model once per server process and score every dataset record."""
    bundle = train_model()
    res = analyze_records(bundle, bundle.data[bundle.feature_names])
    res["reference_label"] = bundle.data["diagnosis"]
    res["data_split"] = np.where(res.index.isin(bundle.test_index), "Held-out test", "Training")
    return bundle, res


def priority_counts(res: pd.DataFrame) -> dict[str, int]:
    vc = res["review_priority"].value_counts()
    return {k: int(vc.get(k, 0)) for k in PRIORITY_ORDER}


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="ms-brand"><img src="{LOGO_URI}" alt="logo"><b>MediScan AI</b></div>'
            '<div class="ms-side-note">Intelligent Clinical Decision Support</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.radio("Navigation", list(PAGES), key="nav_page", label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            '<div class="ms-side-note"><b>Educational prototype.</b> Not a medical device. '
            "Not for diagnosing real patients. Do not upload identifiable patient data.</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# Page 1 - Home Dashboard
# =============================================================================
def page_home(bundle, results) -> None:
    m, q = bundle.metrics, bundle.quality
    page_header("MediScan AI", "Intelligent Clinical Decision Support System")

    with st.container(border=True):
        st.markdown(
            "### Welcome 👋\n"
            "MediScan AI shows how machine learning can **help healthcare professionals prioritise cases for further "
            "review**. It analyses a public educational dataset (Wisconsin Diagnostic Breast Cancer), estimates the "
            "probability of each class, groups cases into *illustrative* review-priority categories and explains which "
            "measurements influenced the estimate."
        )
    st.markdown(banner(DISCLAIMER_SHORT, "warn"), unsafe_allow_html=True)

    cards_row([
        ("Total dataset records", f"{q['n_records']:,}", "Wisconsin Diagnostic Breast Cancer"),
        ("Features per record", f"{q['n_features']}", "Cell-nucleus measurements"),
        ("Train / test split", f"{m['n_train']} / {m['n_test']}", f"Stratified, seed {RANDOM_SEED}", "#1D4E89"),
        ("5-fold CV ROC-AUC", f"{bundle.cv_auc_mean:.3f}", f"± {bundle.cv_auc_std:.3f} (training data only)", "#0B2545"),
    ])
    st.write("")
    st.markdown(f"##### Model evaluation on the held-out test set ({m['n_test']} records)")
    cards_row([
        ("Accuracy", f"{m['accuracy']:.1%}", "Share of correct predictions"),
        ("Precision", f"{m['precision']:.1%}", "Of cases flagged malignant, how many were", "#1D4E89"),
        ("Recall", f"{m['recall']:.1%}", "Of malignant cases, how many were caught", "#1D4E89"),
        ("F1-score", f"{m['f1']:.1%}", "Balance of precision and recall"),
        ("ROC-AUC", f"{m['roc_auc']:.3f}", "Ability to rank malignant above benign", "#0B2545"),
    ])
    st.caption("Precision, recall and F1 treat *malignant* as the positive class. "
               "Scores are for this small public dataset only - not real-world diagnostic accuracy.")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.distribution_donut(q["class_counts"]), "home_donut")
    with c2:
        show_chart(charts.priority_bar(priority_counts(results)), "home_priority")
    c3, c4 = st.columns(2)
    with c3:
        show_chart(charts.metrics_bar(m), "home_metrics")
    with c4:
        show_chart(charts.roc_fig(bundle.roc["fpr"], bundle.roc["tpr"], m["roc_auc"]), "home_roc")

    st.markdown("#### Explore")
    n1, n2, n3, n4 = st.columns(4)
    n1.button("🔬 Analyse a case", on_click=go, args=(PAGE_CASE,), key="nav_case")
    n2.button("🗂️ Explore the dataset", on_click=go, args=(PAGE_DATA,), key="nav_data")
    n3.button("🧠 Model insights", on_click=go, args=(PAGE_MODEL,), key="nav_model")
    n4.button("🛡️ Safety & limitations", on_click=go, args=(PAGE_ABOUT,), key="nav_about")


# =============================================================================
# Page 2 - Case Analysis
# =============================================================================
def narrate(contrib: pd.DataFrame, n: int = 3) -> str:
    """Plain-English summary of the strongest drivers of the estimate."""
    lines = []
    for _, r in contrib.head(n).iterrows():
        side = "above" if r["z_score"] > 0 else "below"
        toward = "toward the **malignant** class" if r["contribution"] > 0 else "toward the **benign** class"
        lines.append(f"- **{r['feature']}** is {abs(r['z_score']):.1f} standard deviations {side} the training average, "
                     f"which pushes the estimate {toward}.")
    return "\n".join(lines)


def page_case(bundle, results) -> None:
    page_header("Case Analysis", "Review one record: probabilities, illustrative priority and the features behind them")
    st.markdown(banner(DISCLAIMER_SHORT, "warn"), unsafe_allow_html=True)

    BUILTIN, UPLOADED = "Built-in dataset", "Uploaded batch"
    uploaded = st.session_state.get("uploaded")
    source = BUILTIN
    if uploaded is not None:
        source = st.radio("Case source", [BUILTIN, UPLOADED], horizontal=True, key="case_source")

    if source == UPLOADED and uploaded is not None:
        res_df, feat_df, pick_key = uploaded["results"], uploaded["features"], "case_pick_upload"
    else:
        source, res_df, feat_df, pick_key = BUILTIN, results, bundle.data[bundle.feature_names], "case_pick_builtin"

    n = len(res_df)
    st.session_state.setdefault(pick_key, 0)
    if st.session_state[pick_key] >= n:
        st.session_state[pick_key] = 0

    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        pos = st.selectbox("Select a case", options=list(range(n)), key=pick_key,
                           format_func=lambda i: f"Case #{res_df['case_id'].iloc[i]}")
    with c_btn:
        st.write("")
        st.button("🎲 Random case", on_click=pick_random, args=(pick_key, n), key="rand_case")

    row = res_df.iloc[pos]
    feats = feat_df.iloc[pos]
    p = float(row["p_malignant"])
    pr = priority_for(p)

    if source == BUILTIN:
        ref, split = row["reference_label"], row["data_split"]
        st.markdown(f"**Dataset reference label:** {ref} &nbsp;·&nbsp; **Data split:** {split}")
        if split == "Training":
            st.warning("This record was part of the model's **training data**, so its estimate looks better than it "
                       "would on new cases. Choose a *Held-out test* record (see Dataset Explorer) for a fairer example.")
        if ref != row["model_prediction"]:
            st.error(f"The model's prediction (**{row['model_prediction']}**) differs from the dataset's reference label "
                     f"(**{ref}**). Models make mistakes - this is exactly why human review is essential.")
    elif int(row["features_outside_training_range"]) > 0:
        st.warning(f"{int(row['features_outside_training_range'])} feature value(s) in this record lie outside the range "
                   "seen during training, so the estimate may be unreliable.")

    # --- probability + priority ------------------------------------------------
    left, right = st.columns(2)
    with left:
        show_chart(charts.probability_gauge(p), "case_gauge")
    with right:
        st.markdown("##### Illustrative review-priority category")
        st.markdown(badge(pr.label, pr.color), unsafe_allow_html=True)
        st.write("")
        st.write(pr.action)
        a, b, c = st.columns(3)
        a.metric("P(malignant)", f"{p:.1%}")
        b.metric("P(benign)", f"{1 - p:.1%}")
        c.metric("Model prediction", row["model_prediction"])
        st.caption(f"Illustrative thresholds: ≥ {HIGH_THRESHOLD:.0%} = Priority 1, ≥ {MEDIUM_THRESHOLD:.0%} = Priority 2, "
                   "otherwise Priority 3. Prediction label uses a 50 % cut-off. Not clinically validated.")

    # --- record information -----------------------------------------------------
    contrib = explain_case(bundle, feats)
    st.markdown("#### Selected record")
    info = contrib.set_index("feature").loc[bundle.feature_names].reset_index()
    info = info.rename(columns={"feature": "Feature", "value": "Value", "training_mean": "Training average",
                                "z_score": "Std devs from average"})[["Feature", "Value", "Training average", "Std devs from average"]]
    mean_rows = info[info["Feature"].str.startswith("mean ")].round(3)
    show_table(mean_rows)
    with st.expander("Show all 30 features"):
        show_table(info.round(3), height=420)

    # --- explanation --------------------------------------------------------------
    st.markdown("#### Why did the model estimate this?")
    e1, e2 = st.columns([3, 2])
    with e1:
        show_chart(charts.contribution_fig(contrib), "case_contrib")
    with e2:
        st.markdown("**Strongest drivers**")
        st.markdown(narrate(contrib))
        st.caption("Contributions come directly from the Logistic Regression coefficients, so they describe how "
                   "*this model* reached its estimate - they do not prove a medical cause.")

    # --- human review recommendation ------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "### ✅ Human-review recommendation\n"
            f"**Suggested action:** {pr.action}\n\n"
            "- This estimate is **decision support only** and is not a diagnosis.\n"
            "- A qualified healthcare professional should review the underlying clinical information and decide on "
            "any further assessment.\n"
            "- Do not rely on a low score to dismiss concern, or on a high score alone to conclude disease."
        )


# =============================================================================
# Page 3 - Dataset Explorer
# =============================================================================
def tab_browse(bundle, results) -> None:
    full = results.join(bundle.data[bundle.feature_names])
    c1, c2, c3, c4 = st.columns(4)
    query = c1.text_input("Search case number", placeholder="e.g. 12", key="ds_query")
    labels = c2.multiselect("Reference label", ["Benign", "Malignant"], default=["Benign", "Malignant"], key="ds_labels")
    prios = c3.multiselect("Review priority", PRIORITY_ORDER, default=PRIORITY_ORDER, key="ds_prios")
    splits = c4.multiselect("Data split", ["Training", "Held-out test"], default=["Training", "Held-out test"], key="ds_splits")

    view = full[full["reference_label"].isin(labels) & full["review_priority"].isin(prios) & full["data_split"].isin(splits)]
    if query.strip():
        view = view[view["case_id"].astype(str).str.contains(query.strip(), regex=False)]

    with st.expander("Filter by a feature value"):
        feat = st.selectbox("Feature", bundle.feature_names, key="ds_feat")
        lo, hi = float(full[feat].min()), float(full[feat].max())
        lo_sel, hi_sel = st.slider("Range", lo, hi, (lo, hi), step=(hi - lo) / 200, format="%.4g", key=f"ds_rng_{feat}")
        view = view[(view[feat] >= lo_sel) & (view[feat] <= hi_sel)]

    show_all = st.checkbox("Show all 30 feature columns", value=False, key="ds_all")
    st.caption(f"Showing **{len(view):,}** of {len(full):,} records.")
    if view.empty:
        st.info("No records match the current filters. Try removing a filter or widening the feature range.")
        return

    table = pd.DataFrame({
        "Case": view["case_id"], "Reference label": view["reference_label"], "Data split": view["data_split"],
        "P(malignant) %": (view["p_malignant"] * 100).round(1), "Model prediction": view["model_prediction"],
        "Review priority": view["review_priority"],
    })
    if show_all:
        table = pd.concat([table, view[bundle.feature_names].round(4)], axis=1)
    show_table(table, height=430)
    st.download_button("⬇️ Download these results (CSV)", data=report_csv(view), file_name="mediscan_dataset_results.csv",
                       mime="text/csv", key="dl_browse")


def tab_stats(bundle, results) -> None:
    st.markdown("#### Category distribution")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.distribution_donut(bundle.quality["class_counts"], "Reference labels in the dataset"), "ds_donut")
    with c2:
        show_chart(charts.priority_bar(priority_counts(results)), "ds_priority")

    st.markdown("#### Feature statistics")
    stats = bundle.data[bundle.feature_names].describe().T.round(4)
    stats.insert(0, "feature", stats.index)
    show_table(stats.reset_index(drop=True), height=420)
    st.download_button("⬇️ Download feature statistics (CSV)", data=stats.to_csv(index=False).encode("utf-8"),
                       file_name="mediscan_feature_statistics.csv", mime="text/csv", key="dl_stats")

    st.markdown("#### Compare a feature across classes")
    feat = st.selectbox("Feature", bundle.feature_names, key="stat_feat")
    show_chart(charts.class_histogram(bundle.data, feat), "ds_hist")
    by_class = bundle.data.groupby("diagnosis")[feat].agg(["count", "mean", "std", "min", "max"]).round(4).reset_index()
    show_table(by_class.rename(columns={"diagnosis": "Class"}))


def tab_upload(bundle) -> None:
    st.markdown("#### Batch analysis of your own CSV")
    st.markdown(banner(
        "<b>Privacy first.</b> Use only public, synthetic or fully anonymised data. Files with columns such as name, "
        "patient ID, address, phone, email or date of birth are rejected. Uploaded data is processed in memory for this "
        "session only - it is never saved to disk or to a database.", "safe"), unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV template (headers only)", data=template_csv_bytes(), file_name="mediscan_template.csv",
                       mime="text/csv", key="dl_template")
    d2.download_button("⬇️ Download example file (10 public test records)", data=sample_csv_bytes(bundle),
                       file_name="mediscan_example_upload.csv", mime="text/csv", key="dl_example")
    st.caption(f"Required: the 30 numeric feature columns (optional anonymous `case_id` column). "
               f"Limits: {MAX_UPLOAD_MB} MB and {MAX_ROWS:,} rows. Kaggle-style names such as `radius_mean` are accepted.")

    confirm = st.checkbox("I confirm this file contains only public, synthetic or fully anonymised data - "
                          "no identifiable patient information.", key="privacy_ok")
    nonce = st.session_state.get("upload_nonce", 0)
    file = st.file_uploader("Upload a CSV file", type=["csv"], disabled=not confirm, key=f"uploader_{nonce}")

    if file is not None and confirm:
        with st.spinner("Validating and analysing your file..."):
            check = validate_uploaded_csv(file.getvalue())
            if check.ok:
                res = analyze_records(bundle, check.features, check.case_ids.to_numpy())
                st.session_state["uploaded"] = {"results": res, "features": check.features,
                                                "rows_total": check.rows_total, "rows_valid": check.rows_valid}
            else:
                st.session_state.pop("uploaded", None)
        for msg in check.errors:
            st.error(msg)
        for msg in check.warnings:
            st.warning(msg)
        if check.ok:
            st.success(f"File accepted: {check.rows_valid} of {check.rows_total} rows analysed.")

    data = st.session_state.get("uploaded")
    if data is None:
        if file is None:
            st.info("No file analysed yet. Tick the confirmation box, then upload a CSV (try the example file above).")
        return

    res = data["results"]
    counts = priority_counts(res)
    cards_row([
        ("Rows analysed", f"{data['rows_valid']:,}", f"of {data['rows_total']:,} in the file", "#1D4E89"),
        ("Priority 1 - High", str(counts[PRIORITY_ORDER[0]]), "", PRIORITY_COLORS[PRIORITY_ORDER[0]]),
        ("Priority 2 - Medium", str(counts[PRIORITY_ORDER[1]]), "", PRIORITY_COLORS[PRIORITY_ORDER[1]]),
        ("Priority 3 - Routine", str(counts[PRIORITY_ORDER[2]]), "", PRIORITY_COLORS[PRIORITY_ORDER[2]]),
    ])
    st.write("")
    show_chart(charts.priority_bar(counts, "Review-priority groups in your upload"), "up_priority")
    table = pd.DataFrame({
        "Case": res["case_id"], "P(malignant) %": (res["p_malignant"] * 100).round(1),
        "Model prediction": res["model_prediction"], "Review priority": res["review_priority"],
        "Features outside training range": res["features_outside_training_range"],
    }).sort_values("P(malignant) %", ascending=False)
    show_table(table, height=360)
    b1, b2, b3 = st.columns(3)
    b1.download_button("⬇️ Download report (CSV)", data=report_csv(res), file_name="mediscan_batch_report.csv",
                       mime="text/csv", key="dl_report")
    b2.button("🔬 Open in Case Analysis", on_click=go, args=(PAGE_CASE,), key="up_to_case")
    b3.button("🗑️ Clear uploaded data", on_click=clear_upload, key="up_clear")


def page_data(bundle, results) -> None:
    page_header("Dataset Explorer", "Browse, filter and summarise records - or analyse your own CSV")
    t1, t2, t3 = st.tabs(["📋 Browse & search", "📊 Statistics", "⬆️ Batch upload"])
    with t1:
        tab_browse(bundle, results)
    with t2:
        tab_stats(bundle, results)
    with t3:
        tab_upload(bundle)


# =============================================================================
# Page 4 - AI Model Insights
# =============================================================================
def page_model(bundle, results) -> None:
    m = bundle.metrics
    page_header("AI Model Insights", "How the model performs, what it relies on and where it can fail")
    st.markdown(banner(
        f"Metrics are measured on <b>{m['n_test']} held-out test records</b> the model never saw during training. "
        "They describe this small public dataset - not real-world clinical performance.", "info"), unsafe_allow_html=True)

    table = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (malignant)", "Recall / sensitivity (malignant)", "F1-score (malignant)",
                   "ROC-AUC", "Specificity (benign)", "5-fold CV ROC-AUC (training data)"],
        "Value": [f"{m['accuracy']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                  f"{m['roc_auc']:.3f}", f"{m['specificity']:.3f}", f"{bundle.cv_auc_mean:.3f} ± {bundle.cv_auc_std:.3f}"],
        "Plain-English meaning": [
            "Share of all test records classified correctly.",
            "When the model says 'malignant', how often it is right.",
            "Share of truly malignant records the model caught.",
            "Single score balancing precision and recall.",
            "Chance that a random malignant record scores higher than a random benign one (1.0 = perfect).",
            "Share of truly benign records correctly identified.",
            "Stability check using only the training split (no test data involved).",
        ],
    })
    show_table(table)

    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.confusion_matrix_fig(bundle.confusion), "mi_cm")
        st.caption(f"Missed malignant cases (false negatives): **{m['false_negatives']}** · "
                   f"benign flagged as malignant (false positives): **{m['false_positives']}**. "
                   "Missed malignant cases are the costliest error in a screening setting.")
    with c2:
        show_chart(charts.roc_fig(bundle.roc["fpr"], bundle.roc["tpr"], m["roc_auc"]), "mi_roc")
    c3, c4 = st.columns(2)
    with c3:
        show_chart(charts.probability_histogram(bundle.test_probs), "mi_probs")
    with c4:
        show_chart(charts.metrics_bar(m), "mi_metrics")

    st.markdown("#### Feature importance")
    show_chart(charts.importance_fig(bundle.coefficients), "mi_importance")
    st.caption("Features are standardised first, so coefficients are comparable. Many WDBC features are strongly correlated "
               "(e.g. radius, perimeter and area), so credit can be shared between them - read importance as "
               "*how this model weighs inputs*, not as proof of biological causes.")
    with st.expander("Show all coefficients"):
        coef = bundle.coefficients[["feature", "malignant_effect", "direction"]].round(3)
        show_table(coef.rename(columns={"feature": "Feature", "malignant_effect": "Effect on malignant log-odds",
                                        "direction": "Direction"}), height=420)

    with st.container(border=True):
        st.markdown(
            "### ⚠️ Model limitations\n"
            "- **Small, clean, public dataset** (569 records from one source) - real clinical data is messier and more varied.\n"
            "- **Not clinically validated.** No external validation, no prospective study, no regulatory approval.\n"
            "- **Linear model:** Logistic Regression cannot capture complex interactions between measurements.\n"
            "- **Tabular measurements only:** it does not look at images, patient history, symptoms or other context.\n"
            "- **Possible bias:** the data does not describe who the patients were, so performance across different "
            "populations is unknown.\n"
            "- **Probabilities are estimates** and may not be perfectly calibrated.\n"
            "- **Illustrative priority thresholds** were chosen for demonstration, not derived from clinical evidence.\n"
            "- A small number of errors in a few hundred records can still be significant in practice."
        )


# =============================================================================
# Page 5 - About & Safety
# =============================================================================
def page_about(bundle, results) -> None:
    q = bundle.quality
    page_header("About & Safety", "Purpose, technology, data, privacy and responsible use")
    st.markdown(banner(DISCLAIMER_SHORT, "warn"), unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown(
                "### 🎯 Project purpose\n"
                "MediScan AI answers the Web-A-Thon challenge: *develop a decision-support tool that analyses approved "
                "medical images or clinical datasets to identify potential signs of disease and help healthcare "
                "professionals prioritise cases for further assessment.*\n\n"
                "It is a **learning and demonstration prototype** that shows how model outputs, probabilities and "
                "explanations can support - never replace - professional judgement."
            )
        with st.container(border=True):
            st.markdown(
                "### 🛠️ Technologies used\n"
                "- **Python** - programming language\n- **Streamlit** - web interface\n"
                "- **Pandas & NumPy** - data processing\n- **Scikit-learn** - preprocessing and Logistic Regression\n"
                "- **Plotly** - interactive charts\n- **GitHub / Streamlit Community Cloud** - version control and hosting"
            )
        with st.container(border=True):
            st.markdown(
                "### 📚 Dataset source\n"
                f"**Breast Cancer Wisconsin (Diagnostic)** dataset - {q['n_records']} records, {q['n_features']} numeric "
                "features computed from digitised images of fine-needle-aspirate samples of breast masses "
                "(created by W. H. Wolberg, W. N. Street and O. L. Mangasarian; available from the UCI Machine Learning "
                "Repository and bundled with scikit-learn). Labels: "
                + ", ".join(f"{k} = {v}" for k, v in q["class_counts"].items()) + ".\n\n"
                "Cleaning checks run at start-up on missing values, duplicate rows, negative measurements and label values "
                f"(found: {q['missing_values_found']} missing values, {q['duplicate_rows_found']} duplicate rows, "
                f"{q['negative_values_found']} negative values)."
            )
    with b:
        with st.container(border=True):
            st.markdown(
                "### ⚙️ How the model works\n"
                f"1. The data is split into **training ({100 - int(TEST_SIZE * 100)} %)** and **test ({int(TEST_SIZE * 100)} %)** sets "
                f"with a fixed random seed ({RANDOM_SEED}) so results are reproducible.\n"
                "2. A **StandardScaler** puts all features on the same scale - fitted on training data only to avoid data leakage.\n"
                "3. **Logistic Regression** learns how each feature shifts the odds of the malignant class and outputs a probability.\n"
                "4. The model is scored on the unseen test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).\n"
                "5. Probabilities are mapped to **illustrative** priority groups, each paired with a human-review recommendation.\n"
                "6. Explanations show each feature's exact contribution to the model's log-odds."
            )
        with st.container(border=True):
            st.markdown(
                "### 🔒 Privacy considerations\n"
                "- The built-in data is public and anonymous.\n"
                "- Uploaded CSVs are processed **in memory only** - never written to disk or a database by this app.\n"
                "- Files with name-, ID-, contact- or birth-date-style columns are **rejected**.\n"
                "- Please **do not upload real patient information**, even if you believe it is de-identified.\n"
                "- Uploaded data disappears when you press *Clear uploaded data* or the session ends.\n"
                "- Streamlit Community Cloud hosts the app on third-party servers - another reason to use public or synthetic data only."
            )
        with st.container(border=True):
            st.markdown(
                "### ⚖️ Limitations & responsible AI use\n"
                "- Outputs are **estimates**, not diagnoses, and must be reviewed by qualified professionals.\n"
                "- No clinical validation, regulatory approval or real-world accuracy is claimed.\n"
                "- Priority categories are illustrative and not a triage protocol.\n"
                "- Low scores do not rule out disease; high scores do not confirm it.\n"
                "- Training records show optimistic results - always prefer held-out records for demonstrations.\n"
                "- Real deployment would need external validation, bias audits, security review and regulatory clearance."
            )
    st.button("← Back to dashboard", on_click=go, args=(PAGE_HOME,), key="about_home")


# =============================================================================
# Router
# =============================================================================
PAGES = {
    PAGE_HOME: page_home,
    PAGE_CASE: page_case,
    PAGE_DATA: page_data,
    PAGE_MODEL: page_model,
    PAGE_ABOUT: page_about,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("nav_page", PAGE_HOME)
    if st.session_state["nav_page"] not in PAGES:          # guard against stale state
        st.session_state["nav_page"] = PAGE_HOME
    with st.spinner("Loading dataset and training the model..."):
        bundle, results = get_context()
    render_sidebar()
    PAGES[st.session_state["nav_page"]](bundle, results)


main()
