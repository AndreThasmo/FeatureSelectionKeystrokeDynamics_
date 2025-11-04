import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import ast
from pathlib import Path
import io

# Matplotlib/Seaborn global style
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def fig3_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def _feature_sort_key(name):
    if isinstance(name, str) and re.fullmatch(r"f\d+", name):
        return (0, int(name[1:]))
    return (1, str(name))


# Helper: robustly parse a "Selected Features" cell to list of feature names
def _parse_sel_features(val):
    """Parse 'Selected Features' cell into a clean list of feature names.
    Accepts comma-separated strings ("f1, f2") or Python-list-like strings ("['f1','f2']").
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "none":
        return []
    # Try to parse as Python literal list/tuple first
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip("'\"") for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Fallback: split by commas with optional spaces
    parts = re.split(r"\s*,\s*", s)
    return [p.strip().strip("'\"[]") for p in parts if p and p.lower() != "none"]


# Set seaborn theme
sns.set_theme(style="whitegrid", palette="deep", context="notebook")

# Configurações gerais
st.set_page_config(
    page_title="Keystroke Dynamics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Theme toggle (Light/Dark) and dynamic styling
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.session_state.dark_mode = st.sidebar.toggle("Dark mode", value=st.session_state.dark_mode)


def apply_theme(dark: bool) -> None:
    """Apply plotting styles and CSS for light/dark modes."""
    # Keep previously set font sizes; only update colors/backgrounds
    if dark:
        sns.set_theme(style="darkgrid", palette="deep", context="notebook")
        plt.rcParams.update({
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "text.color": "#e0e0e0",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0",
        })
        css = """
        <style>
        body { color: #e0e0e0; background-color: #0e1117; }
        .block-container { max-width: 1200px; padding-top: 0rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background-color: #111827; }
        h1, h2, h3 { font-weight: 600; color: #e0e0e0; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; color: #e0e0e0; }
        </style>
        """
    else:
        sns.set_theme(style="whitegrid", palette="deep", context="notebook")
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        })
        css = """
        <style>
        .block-container { max-width: 1200px; padding-top: 0rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background-color: #f8f9fb; }
        h1, h2, h3 { font-weight: 600; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


apply_theme(st.session_state.dark_mode)

# Header with UFABC logo and formal subtitle (logo hosted on Wikimedia Commons)
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/0/03/Logo_UFABC.png"
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image(LOGO_URL, use_container_width=False, width=90)
with col_title:
    st.title("Keystroke Dynamics Evaluation Dashboard")
    st.markdown(
        "<div style='color:#5f6368; font-size: 0.95rem;'>Universidade Federal do ABC — Iniciação Científica</div>",
        unsafe_allow_html=True,
    )
st.markdown("---")

st.caption("Dica: se a barra lateral não aparecer, use a seta no canto superior esquerdo (☰) para abrir.")


# Função para carregar o CSV mais recente automaticamente
@st.cache_data
def load_latest_csv():
    output_dir = Path("data/output")
    files = list(output_dir.glob("*.csv"))
    if not files:
        return None, None
    latest = max(files, key=lambda f: f.stat().st_ctime)
    df = pd.read_csv(latest)
    return df, latest.name


# Upload manual ou seleção automática
st.sidebar.header("Load Data")
upload = st.sidebar.file_uploader("Upload CSV manually", type=["csv"]) 
if upload:
    df = pd.read_csv(upload)
    file_name = upload.name
else:
    df, file_name = load_latest_csv()
    if df is not None:
        st.sidebar.markdown(f"**Auto-loaded:** `{file_name}`")

if df is None:
    st.warning("No CSV file found in `data/output/` and nothing uploaded.")
    st.stop()

st.sidebar.divider()

st.sidebar.header("Filters")


# Optional fallback for rows without selected features
if "Selected Features" in df.columns:
    # Normalize empties / placeholders to NA
    df["Selected Features"] = df["Selected Features"].astype("string").str.strip()
    df["Selected Features"] = df["Selected Features"].replace({"": pd.NA, "[]": pd.NA, "nan": pd.NA, "None": pd.NA})
    with st.sidebar.expander("Selected Features Fallback", expanded=False):
        fallback_input = st.text_area(
            "Default features (comma-separated) for empty rows",
            placeholder="f1, f2, f3"
        )
        apply_fallback = st.checkbox("Apply fallback to empty 'Selected Features'", value=False)
    if apply_fallback and isinstance(fallback_input, str) and fallback_input.strip():
        fallback_norm = ", ".join([s.strip() for s in fallback_input.split(",") if s.strip()])
        mask_empty_sel = df["Selected Features"].isna()
        df.loc[mask_empty_sel, "Selected Features"] = fallback_norm

# Após aplicar o fallback para Selected Features (ou logo após carregar o CSV)

# Garante que entradas nulas de 'Feature Selection Algorithm' sejam substituídas por "None"
if "Feature Selection Algorithm" in df.columns:
    df["Feature Selection Algorithm"] = df["Feature Selection Algorithm"].fillna("None")
    
# Filtros dinâmicos: só aparecem se a coluna existir
selected_algo = None
if "Feature Selection Algorithm" in df.columns:
    algorithms = df["Feature Selection Algorithm"].dropna().astype(str).unique().tolist()
    selected_algo = st.sidebar.multiselect("Feature Selection Algorithm", algorithms, default=algorithms)

with st.sidebar.expander("User Filters", expanded=False):
    selected_users = None
    select_all_users = False
    if "User" in df.columns:
        users = sorted(df["User"].dropna().astype(str).unique().tolist())
        selected_users = st.multiselect("User", users, default=users)
        select_all_users = st.checkbox("Select All Users", value=True)
        if select_all_users:
            selected_users = users
    else:
        st.write("No 'User' column in data.")

selected_n = None
if "Number of Features" in df.columns:
    n_features = sorted(pd.to_numeric(df["Number of Features"], errors="coerce").dropna().unique().tolist())
    selected_n = st.sidebar.multiselect("Number of Features", n_features, default=n_features)

selected_classifiers = None
if "Classifier" in df.columns:
    classifiers = df["Classifier"].dropna().astype(str).unique().tolist()
    selected_classifiers = st.sidebar.multiselect("Classifier", classifiers, default=classifiers)

selected_seeds = None
if "random_state" in df.columns:
    random_states = sorted(pd.to_numeric(df["random_state"], errors="coerce").dropna().unique().tolist())
    selected_seeds = st.sidebar.multiselect("Random State", random_states, default=random_states)

# Filtrar dados
filtered_df = df.copy()
if selected_algo is not None and "Feature Selection Algorithm" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Feature Selection Algorithm"].astype(str).isin(selected_algo)]
if selected_users is not None and "User" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["User"].astype(str).isin(selected_users)]
if selected_n is not None and "Number of Features" in filtered_df.columns:
    filtered_df = filtered_df[pd.to_numeric(filtered_df["Number of Features"], errors="coerce").isin(selected_n)]
if selected_classifiers is not None and "Classifier" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Classifier"].astype(str).isin(selected_classifiers)]
if selected_seeds is not None and "random_state" in filtered_df.columns:
    filtered_df = filtered_df[pd.to_numeric(filtered_df["random_state"], errors="coerce").isin(selected_seeds)]


# Adiciona flags de alerta para FPR, FNR e MCC (booleans formais)

def add_alert_flags(df):
    df = df.copy()
    for col in ["FPR", "FNR", "MCC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Coerce FNMR as numeric if present
    if "FNMR" in df.columns:
        df["FNMR"] = pd.to_numeric(df["FNMR"], errors="coerce")
    if "FPR" in df.columns:
        df["FPR High"] = df["FPR"] > 0.30
    if "FNR" in df.columns:
        df["FNR High"] = df["FNR"] > 0.30
    if "FNMR" in df.columns:
        df["FNMR High"] = df["FNMR"] > 0.30
    if "MCC" in df.columns:
        df["MCC Low"] = df["MCC"] < 0.20
    return df


filtered_df = add_alert_flags(filtered_df)

# Métricas globais resumidas fixas no topo
with st.container():
    st.markdown("#### Summary Metrics")
    col_metrics = {
        "Accuracy": "Accuracy",
        "Balanced Accuracy": "Balanced Accuracy",
        "F1-Score": "F1 Score",
        "FPR": "FMR (FPR)",
        "FNMR": "FNMR",
        "MCC": "MCC",
    }

    summary_cols = st.columns(len(col_metrics))

    for i, (col_name, label) in enumerate(col_metrics.items()):
        if col_name in filtered_df.columns:
            series = pd.to_numeric(filtered_df[col_name], errors="coerce")
            value = series.mean()
            summary_cols[i].metric(label, "—" if pd.isna(value) else f"{value:.3f}")
        else:
            summary_cols[i].metric(label, "—")

# Gráficos e tabelas em abas

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Metrics", "Correlation", "Table", "Heatmaps", "Confusion"])

with tab1:
    st.subheader("Balanced Accuracy by Algorithm")
    req_cols = {"Feature Selection Algorithm", "Balanced Accuracy", "Classifier"}
    if not filtered_df.empty and req_cols.issubset(filtered_df.columns):
        plot_df = filtered_df.copy()
        plot_df["Balanced Accuracy"] = pd.to_numeric(plot_df["Balanced Accuracy"], errors="coerce")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=plot_df,
            x="Feature Selection Algorithm",
            y="Balanced Accuracy",
            hue="Classifier",
            ax=ax1,
        )
        # Remove chart title
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel("")
        ax1.set_ylabel("Acurácia balanceada")
        st.markdown("###", unsafe_allow_html=True)
        st.image(fig3_to_image(fig1), use_container_width=False, output_format="PNG")
        plt.close(fig1)
    else:
        st.info("Not enough data/columns to plot Balanced Accuracy by Algorithm.")

    st.subheader("F1-Score by Algorithm")
    req_cols = {"Feature Selection Algorithm", "F1-Score", "Classifier"}
    if not filtered_df.empty and req_cols.issubset(filtered_df.columns):
        plot_df = filtered_df.copy()
        plot_df["F1-Score"] = pd.to_numeric(plot_df["F1-Score"], errors="coerce")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=plot_df,
            x="Feature Selection Algorithm",
            y="F1-Score",
            hue="Classifier",
            ax=ax2,
        )
        # Remove chart title
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel("")
        ax2.set_ylabel("F1-Score")
        st.markdown("###", unsafe_allow_html=True)
        st.image(fig3_to_image(fig2), use_container_width=False, output_format="PNG")
        plt.close(fig2)
    else:
        st.info("Not enough data/columns to plot F1-Score by Algorithm.")

with tab2:
    col_metrics1, col_metrics2 = st.columns(2)

    with col_metrics1:
        st.subheader("Training Time vs Balanced Accuracy")
        req_cols = {"Train Time (s)", "Balanced Accuracy", "Classifier", "Number of Features"}
        if not filtered_df.empty and req_cols.issubset(filtered_df.columns):
            plot_df = filtered_df.copy()
            plot_df["Train Time (s)"] = pd.to_numeric(plot_df["Train Time (s)"], errors="coerce")
            plot_df["Balanced Accuracy"] = pd.to_numeric(plot_df["Balanced Accuracy"], errors="coerce")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(
                data=plot_df,
                x="Train Time (s)",
                y="Balanced Accuracy",
                hue="Classifier",
                style="Number of Features",
                ax=ax3,
            )
            # Remove chart title
            ax3.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax3.set_xlabel("Tempo de treinamento (s)")
            ax3.set_ylabel("Acurácia balanceada")
            st.markdown("###", unsafe_allow_html=True)
            st.image(fig3_to_image(fig3), use_container_width=False, output_format="PNG")
            plt.close(fig3)
        else:
            st.info("Not enough data/columns to plot Training Time vs Balanced Accuracy.")

    with col_metrics2:
        st.subheader("Precision & Recall by User")
        req_cols = {"User", "Precision", "Recall"}
        if not filtered_df.empty and req_cols.issubset(filtered_df.columns):
            agg = filtered_df.copy()
            agg["Precision"] = pd.to_numeric(agg["Precision"], errors="coerce")
            agg["Recall"] = pd.to_numeric(agg["Recall"], errors="coerce")
            agg = (
                agg.groupby("User")[
                    ["Precision", "Recall"]
                ].mean().reset_index().melt(id_vars="User", var_name="Metric", value_name="Score")
            )
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            sns.barplot(data=agg, x="User", y="Score", hue="Metric", ax=ax4)
            # Remove chart title
            ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax4.set_xlabel("Usuário")
            ax4.set_ylabel("Pontuação")
            ax4.tick_params(axis="x", labelrotation=90)
            st.markdown("###", unsafe_allow_html=True)
            st.image(fig3_to_image(fig4), use_container_width=False, output_format="PNG")
            plt.close(fig4)
        else:
            st.info("Not enough data/columns to plot Precision & Recall by User.")

with tab3:
    st.subheader("Correlation")

    # --- Controls ---
    # Available numeric metrics in the current filtered_df
    all_metric_cols = [
        "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score",
        "Specificity", "FPR", "FNR", "FNMR", "MCC", "Train Time (s)", "Test Time (s)",
    ]
    present_metrics = [c for c in all_metric_cols if c in filtered_df.columns]
    # Default selection focuses on evaluation metrics (omit times by default)
    default_metrics = [c for c in [
        "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score",
        "Specificity", "FPR", "FNMR", "MCC",
    ] if c in present_metrics]

    left_ctrl, right_ctrl = st.columns([2, 1])
    with left_ctrl:
        sel_metrics = st.multiselect("Metrics to include in correlation heatmap", options=present_metrics, default=default_metrics)
    with right_ctrl:
        method = st.radio("Method", ["Pearson", "Spearman"], index=0, horizontal=True)
        method = method.lower()

    if len(sel_metrics) < 2:
        st.info("Select at least two metrics to compute correlations.")
    else:
        # --- Global correlation heatmap ---
        num_df = filtered_df[sel_metrics].apply(pd.to_numeric, errors="coerce")
        corr = num_df.corr(method=method)
        fig_c, ax_c = plt.subplots(figsize=(max(6, 0.8*len(sel_metrics)), max(4, 0.6*len(sel_metrics))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax_c)
        # Remove chart title
        st.image(fig3_to_image(fig_c), use_container_width=False, output_format="PNG")
        plt.close(fig_c)

        st.markdown("---")
        # --- Key pairs scatter ---
        st.subheader("Key Pairs — Scatter")
        color_by_candidates = [c for c in ["Feature Selection Algorithm", "Classifier", "User"] if c in filtered_df.columns]
        color_by = st.selectbox("Color points by", color_by_candidates, index=0 if color_by_candidates else None) if color_by_candidates else None

        def _scatter_pair(x_col: str, y_col: str):
            if x_col in filtered_df.columns and y_col in filtered_df.columns:
                plot_df = filtered_df[[x_col, y_col] + ([color_by] if color_by else [])].copy()
                plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
                plot_df = plot_df.dropna(subset=[x_col, y_col])
                # Sample to keep plotting responsive on very large datasets
                if len(plot_df) > 5000:
                    plot_df = plot_df.sample(5000, random_state=42)
                fig_s, ax_s = plt.subplots(figsize=(7, 4))
                if color_by:
                    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=color_by, ax=ax_s, alpha=0.6, s=18)
                    ax_s.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                else:
                    sns.scatterplot(data=plot_df, x=x_col, y=y_col, ax=ax_s, alpha=0.6, s=18)
                # Add simple linear trendline (overall)
                try:
                    x = plot_df[x_col].values
                    y = plot_df[y_col].values
                    if len(x) > 1:
                        coeffs = np.polyfit(x, y, deg=1)
                        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                        y_line = coeffs[0]*x_line + coeffs[1]
                        ax_s.plot(x_line, y_line, linewidth=1)
                except Exception:
                    pass
                # Portuguese axis labels for common metrics
                _pt = {
                    "Accuracy": "Acurácia",
                    "Balanced Accuracy": "Acurácia balanceada",
                    "Precision": "Precisão",
                    "Recall": "Recall",
                    "F1-Score": "F1-Score",
                    "Specificity": "Especificidade",
                    "FPR": "FMR (FPR)",
                    "FNR": "FNR",
                    "FNMR": "FNMR",
                    "MCC": "MCC",
                    "Train Time (s)": "Tempo de treinamento (s)",
                    "Test Time (s)": "Tempo de teste (s)",
                    "User": "Usuário",
                }
                ax_s.set_xlabel(_pt.get(x_col, x_col))
                ax_s.set_ylabel(_pt.get(y_col, y_col))
                st.image(fig3_to_image(fig_s), use_container_width=False, output_format="PNG")
                plt.close(fig_s)
            else:
                st.info(f"Columns '{x_col}' and/or '{y_col}' are not available.")

        cols_pairs = [("F1-Score", "MCC"), ("Precision", "F1-Score")]
        for x_col, y_col in cols_pairs:
            if x_col in present_metrics and y_col in present_metrics:
                _scatter_pair(x_col, y_col)

        st.markdown("---")
        # --- Correlation per Feature Selection Algorithm (optional) ---
        if "Feature Selection Algorithm" in filtered_df.columns:
            show_split = st.checkbox("Show correlation per Feature Selection Algorithm", value=False)
            if show_split:
                algos = sorted(filtered_df["Feature Selection Algorithm"].dropna().astype(str).unique().tolist())
                # fix color scale across sub-heatmaps
                vmin, vmax = -1, 1
                for algo in algos:
                    sub = filtered_df[filtered_df["Feature Selection Algorithm"] == algo]
                    sub_num = sub[sel_metrics].apply(pd.to_numeric, errors="coerce")
                    if sub_num.dropna(how="all").empty:
                        continue
                    corr_a = sub_num.corr(method=method)
                    fig_a, ax_a = plt.subplots(figsize=(max(6, 0.8*len(sel_metrics)), max(4, 0.6*len(sel_metrics))))
                    sns.heatmap(corr_a, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax, ax=ax_a)
                    # Remove chart title
                    st.image(fig3_to_image(fig_a), use_container_width=False, output_format="PNG")
                    plt.close(fig_a)

with tab4:
    st.subheader("Filtered Data Table")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered CSV", csv, "filtered_results.csv", "text/csv")


# Insert new Heatmaps tab with feature occurrence heatmap
with tab5:
    st.subheader("Selected Features — Occurrence Heatmap")
    if "Selected Features" in filtered_df.columns:
        series = filtered_df["Selected Features"].dropna()
        parsed = series.apply(_parse_sel_features)
        flat = [f for sub in parsed for f in sub]
        if len(flat) == 0:
            st.info("No selected features to display.")
        else:
            # Build Feature (Y) × Algorithm (X) occurrence matrix
            if "Feature Selection Algorithm" not in filtered_df.columns:
                st.info("Column 'Feature Selection Algorithm' not found in the data.")
            else:
                tmp = filtered_df[["Feature Selection Algorithm", "Selected Features"]].dropna(subset=["Selected Features"]).copy()
                tmp["__features"] = tmp["Selected Features"].apply(_parse_sel_features)
                exploded = tmp.explode("__features").dropna(subset=["__features"])  # one row per (algo, feature)
                # Count occurrences per (Feature, Algorithm)
                count_pivot = (
                    exploded.groupby(["__features", "Feature Selection Algorithm"])\
                            .size()\
                            .unstack(fill_value=0)
                )
                # Sort rows (features) using custom key and columns (algorithms) alphabetically
                feature_names_sorted = sorted(count_pivot.index.tolist(), key=_feature_sort_key)
                algo_sorted = sorted(count_pivot.columns.astype(str).tolist())
                count_pivot = count_pivot.reindex(index=feature_names_sorted, columns=algo_sorted)

                # Plot: features on Y, algorithms on X
                fig_h, ax_h = plt.subplots(figsize=(max(8, len(algo_sorted) * 1.2), max(4, len(feature_names_sorted) * 0.35)))
                sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax_h, cbar=True, vmin=0)
                # Remove chart title
                ax_h.set_xlabel("Técnica de seleção de atributos")
                ax_h.set_ylabel("Atributo")
                ax_h.set_xticklabels(ax_h.get_xticklabels(), rotation=45, ha="right")
                ax_h.set_yticklabels(ax_h.get_yticklabels(), rotation=0)
                st.image(fig3_to_image(fig_h), use_container_width=True, output_format="PNG")
                plt.close(fig_h)
    else:
        st.info("Column 'Selected Features' not found in the data.")

# Confusion Matrix Viewer tab
with tab6:
    st.subheader("Confusion Matrix Viewer")
    if "Confusion Matrix" in filtered_df.columns:
        # Build a compact key to select a single run
        def _mk_key(row):
            parts = [
                f"User={row.get('User', '—')}",
                f"Alg={row.get('Feature Selection Algorithm', '—')}",
                f"n={row.get('Number of Features', '—')}",
                f"Clf={row.get('Classifier', '—')}",
            ]
            return " | ".join(map(str, parts))
        tmp = filtered_df.copy()
        tmp["__key__"] = tmp.apply(_mk_key, axis=1)
        choice = st.selectbox("Select a run:", tmp["__key__"].tolist())
        row = tmp[tmp["__key__"] == choice].iloc[0]
        try:
            cm = np.array(ast.literal_eval(str(row["Confusion Matrix"])) , dtype=int)
        except Exception:
            cm = None
        if cm is None or cm.shape != (2, 2):
            st.info("Confusion matrix not available or not 2×2 for the selected row.")
        else:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["não_usuário", "usuário"], yticklabels=["não_usuário", "usuário"], ax=ax_cm)
            # Remove chart title
            ax_cm.set_xlabel("Previsto")
            ax_cm.set_ylabel("Real")
            st.image(fig3_to_image(fig_cm), use_container_width=False, output_format="PNG")
            plt.close(fig_cm)
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    else:
        st.info("Column 'Confusion Matrix' not found in the data.")