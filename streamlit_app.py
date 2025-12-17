#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Streamlit dashboard for the equity options IOI + trade dataset.

Run locally with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import warnings
from io import BytesIO
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ----------------------------
# Config
# ----------------------------
IOI_PATH = "data/equity options ioi example 2025-11-15_to_2025-11-19.csv"
TRD_PATH = "data/equity options trades  example 2025-11-15_to_2025-11-19.csv"
DEFAULT_REPORT_NAME = "eq_option_stats.xlsx"

ATM_BAND_PCT = 0.02  # ±2% considered ATM
TENOR_MAP = {"D": 1, "W": 7, "M": 30, "Y": 365}


# ----------------------------
# Helpers
# ----------------------------
def classify_moneyness(row, band: float = ATM_BAND_PCT) -> str:
    m = row.get("moneyness")
    otype = row.get("option_type")
    if pd.isna(m) or pd.isna(otype):
        return "UNK"
    if abs(m) <= band:
        return "ATM"
    if str(otype).upper().startswith("C"):
        return "OTM_call" if m > band else "ITM_call"
    return "OTM_put" if m < -band else "ITM_put"


def _categorize_tenor(days_val):
    """Convert tenor (in days) to a human-readable bucket."""
    if pd.isna(days_val) or days_val < 0:
        return None
    d = float(days_val)
    if d < 7:
        return f"{int(round(d))}D"
    if d < 30:
        return f"{int(round(d / 7))}W"
    if d < 365:
        months = int(round(d / 30))
        if months >= 12:  # cap at 11 months
            return "1Y"
        return f"{months}M"
    years = int(round(d / 365))
    return f"{years}Y"


def _tenor_sort_value(bucket: str) -> float:
    """Numeric days for sorting tenor buckets."""
    if not bucket or pd.isna(bucket):
        return np.inf
    s = str(bucket).strip().upper()
    unit = s[-1]
    try:
        val = float(s[:-1])
    except Exception:
        return np.inf
    if unit == "D":
        return val
    if unit == "W":
        return val * 7
    if unit == "M":
        return val * 30
    if unit == "Y":
        return val * 365
    return np.inf


# ----------------------------
# Load and unify
# ----------------------------
def _read_csv_any(source):
    """Accept a path, buffer, or UploadedFile and return a DataFrame."""
    if source is None:
        return pd.DataFrame()
    if isinstance(source, (str, os.PathLike)):
        source = _resolve_path(source)
    if hasattr(source, "read"):
        source.seek(0)
    return pd.read_csv(source)


def _resolve_path(path_like: str | os.PathLike) -> str:
    """Resolve relative paths against the repo/app directory."""
    path_str = os.fspath(path_like)
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(BASE_DIR, path_str)


def load_inputs(ioi_source=IOI_PATH, trd_source=TRD_PATH):
    ioi = _read_csv_any(ioi_source)
    trd = _read_csv_any(trd_source)

    if ioi.empty and trd.empty:
        raise ValueError("No data found. Provide IOI and trade CSV files.")

    trd["SIDE"] = "T"
    df = pd.concat([ioi, trd], ignore_index=True)

    # Tag message type
    df["SIDE"] = df["SIDE"].astype(str).str.strip().str.upper()
    df["is_ioi"] = df["SIDE"].isin(["B", "O"])
    df["is_trade"] = df["SIDE"].eq("T")
    df["INTEREST_ID"] = df["RECORD_ID"].astype(str).str.split("-").str[0]

    # ----------------------------
    # Timezone correction
    # ----------------------------
    if "TIMESTAMP_UTC" in df.columns:
        df["TIMESTAMP_UTC"] = pd.to_datetime(
            df["TIMESTAMP_UTC"],
            format="%Y-%m-%dT%H:%M:%S",
            errors="raise",
        )
    if "TIMESTAMP_NY" in df.columns:
        df["TIMESTAMP_NY"] = pd.to_datetime(
            df["TIMESTAMP_NY"],
            format="%Y-%m-%dT%H:%M:%S",
            errors="raise",
        )

    df["TIMESTAMP_UTC"] = df["TIMESTAMP_UTC"].dt.tz_localize(None)
    df["TIMESTAMP_NY"] = df["TIMESTAMP_NY"].dt.tz_localize(None)

    # ----------------------------
    # Business date / expiry parsing
    # ----------------------------
    if "BUSINESS_DATE" in df.columns:
        df["BUSINESS_DATE"] = pd.to_datetime(
            df["BUSINESS_DATE"], format="%d-%b-%y", errors="coerce", dayfirst=True
        )
    if "EXPIRY" in df.columns:
        df["EXPIRY"] = pd.to_datetime(df["EXPIRY"], errors="coerce")

    # ----------------------------
    # Numeric cleanup
    # ----------------------------
    for c in ["STRIKE", "PREMIUM", "REF_PRICE", "VOLUME", "DELTA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "UNDERLYING_TYPE" not in df.columns:
        df["UNDERLYING_TYPE"] = "STOCK"

    df["NOTIONAL"] = (
        df.get("VOLUME", 0).fillna(0) * df.get("REF_PRICE", 0).fillna(0) * 100.0
    )

    # Canonical
    colmap = {
        "timestamp": "TIMESTAMP_NY",
        "record_id": "RECORD_ID",
        "underlying": "UNDERLYING",
        "underlying_type": "UNDERLYING_TYPE",
        "option_type": "STRATEGY",
        "expiry": "EXPIRY",
        "strike": "STRIKE",
        "premium": "PREMIUM",
        "spot": "REF_PRICE",
        "size": "VOLUME",
        "delta": "DELTA",
        "status": "STATUS",
        "side": "SIDE",
        "business_date": "BUSINESS_DATE",
        "is_ioi": "is_ioi",
        "is_trade": "is_trade",
        "notional": "NOTIONAL",
        "interest_id": "INTEREST_ID",
    }
    norm = pd.DataFrame({k: df.get(v, np.nan) for k, v in colmap.items()})

    norm["timestamp"] = pd.to_datetime(norm["timestamp"], errors="coerce")
    norm["date"] = pd.to_datetime(norm["timestamp"], errors="coerce").dt.date
    norm["hour"] = pd.to_datetime(norm["timestamp"], errors="coerce").dt.hour

    # ----------------------------
    # Tenor computation
    # ----------------------------
    expiry_dt = pd.to_datetime(df["EXPIRY"], errors="coerce")
    biz_dt = pd.to_datetime(df["BUSINESS_DATE"], errors="coerce")
    df["TENOR_DAYS"] = (expiry_dt - biz_dt).dt.days
    df["TENOR_NORM"] = df["TENOR_DAYS"].apply(
        lambda x: (
            f"{int(round(x/30))}M"
            if pd.notna(x) and 0 < x < 365
            else f"{round(x/365,1)}Y"
            if pd.notna(x) and x >= 365
            else None
        )
    )

    norm["tenor_days"] = df["TENOR_DAYS"]
    norm["tenor_norm"] = df["TENOR_NORM"]

    norm["moneyness"] = (norm["strike"] / norm["spot"]) - 1.0
    norm["moneyness_class"] = norm.apply(classify_moneyness, axis=1)
    norm["prem_norm_spot"] = norm["premium"] / norm["spot"]
    norm["prem_norm_strike"] = norm["premium"] / norm["strike"]

    return df, norm


# ----------------------------
# Diagnostics
# ----------------------------
def field_completeness(df_norm):
    rows = [
        {
            "field": c,
            "non_null_pct": float(df_norm[c].notna().mean()) * 100.0,
            "example_value": df_norm[c].dropna().iloc[0] if df_norm[c].notna().any() else None,
        }
        for c in df_norm.columns
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("non_null_pct", ascending=False)
        .reset_index(drop=True)
    )


def coverage_tables(df_norm):
    df_clean = df_norm.copy()
    df_clean["expiry"] = pd.to_datetime(df_clean["expiry"], errors="coerce")
    df_clean["tenor_norm"] = df_clean["tenor_norm"].replace("", np.nan)

    cov_exp = (
        df_clean[df_clean["expiry"].notna()]
        .groupby(["underlying", "expiry"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["underlying", "expiry"])
    )

    cov_tenor = (
        df_clean[df_clean["tenor_norm"].notna()]
        .groupby(["underlying", "tenor_norm"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["underlying", "tenor_norm"])
    )

    return cov_tenor, cov_exp


# ----------------------------
# Lifecycle (INTEREST_ID)
# ----------------------------
def ioi_trade_match(df_norm):
    if "interest_id" not in df_norm.columns:
        return (
            pd.DataFrame(
                [
                    {
                        "method": "interest_id_link",
                        "match_rate_pct": np.nan,
                        "notes": "INTEREST_ID not found",
                    }
                ]
            ),
            pd.DataFrame(),
        )

    dfv = df_norm.dropna(subset=["interest_id"]).copy()
    dfv["interest_id"] = dfv["interest_id"].astype(str)

    lifecycle = (
        dfv.groupby("interest_id")
        .agg(
            n_iois=("is_ioi", lambda s: int((s == True).sum())),  # noqa: E712
            n_trades=("is_trade", lambda s: int((s == True).sum())),  # noqa: E712
            first_ioi_time=(
                "timestamp",
                lambda s: s[dfv.loc[s.index, "is_ioi"]].min()
                if (dfv.loc[s.index, "is_ioi"]).any()
                else pd.NaT,
            ),
            first_trade_time=(
                "timestamp",
                lambda s: s[dfv.loc[s.index, "is_trade"]].min()
                if (dfv.loc[s.index, "is_trade"]).any()
                else pd.NaT,
            ),
            underlying=("underlying", "first"),
            option_type=("option_type", "first"),
            expiry=("expiry", "first"),
            strike=("strike", "first"),
            total_ioi_vol=("size", lambda s: s[dfv.loc[s.index, "is_ioi"]].sum()),
            total_trade_vol=("size", lambda s: s[dfv.loc[s.index, "is_trade"]].sum()),
        )
        .reset_index()
    )

    lifecycle["has_trade"] = lifecycle["n_trades"] > 0
    lifecycle["latency_sec"] = (
        (lifecycle["first_trade_time"] - lifecycle["first_ioi_time"])
        .dt.total_seconds()
        .where(lifecycle["n_iois"] > 0)
    )
    lifecycle["trade_to_ioi_ratio"] = np.where(
        lifecycle["total_ioi_vol"] > 0,
        lifecycle["total_trade_vol"] / lifecycle["total_ioi_vol"],
        np.nan,
    )

    total_iois = (lifecycle["n_iois"] > 0).sum()
    matched = lifecycle["has_trade"].sum()
    match_rate = 100.0 * matched / max(1, total_iois)

    summary = pd.DataFrame(
        [
            {
                "method": "interest_id_link",
                "match_rate_pct": match_rate,
                "total_interest_ids": lifecycle.shape[0],
                "interest_ids_with_trades": matched,
                "avg_latency_sec": lifecycle["latency_sec"].dropna().mean(),
                "avg_trade_to_ioi_ratio": lifecycle["trade_to_ioi_ratio"].dropna().mean(),
                "notes": "Deterministic lifecycle via INTEREST_ID",
            }
        ]
    )

    return summary, lifecycle.sort_values(["first_ioi_time", "interest_id"])


# ----------------------------
# Simple tables
# ----------------------------
def activity_by_hour(df_norm):
    return (
        df_norm.groupby("hour")
        .agg(
            trade_cnt=("is_trade", lambda s: int((s == True).sum())),  # noqa: E712
            ioi_cnt=("is_ioi", lambda s: int((s == True).sum())),  # noqa: E712
            all_msgs=("timestamp", "count"),
        )
        .reset_index()
    )


def leaders(df_norm):
    return (
        df_norm.groupby("underlying")
        .agg(
            msgs=("timestamp", "count"),
            trades=("is_trade", lambda s: int((s == True).sum())),  # noqa: E712
            iois=("is_ioi", lambda s: int((s == True).sum())),  # noqa: E712
            notional_sum=("notional", "sum"),
        )
        .sort_values(["notional_sum", "msgs"], ascending=False)
        .reset_index()
    )


def buy_sell_imbalance(df_norm, leaders_df):
    df_tmp = df_norm[df_norm["underlying"].isin(leaders_df.underlying)]
    df_ioi = df_tmp[df_tmp["is_ioi"] & df_tmp["side"].isin(["B", "O"])].copy()
    if df_ioi.empty:
        return pd.DataFrame()
    cnt = (
        df_ioi.groupby(["date", "underlying", "side"])["premium"]
        .count()
        .unstack(fill_value=0)
        .rename(columns={"B": "bid_cnt", "O": "offer_cnt"})
    )
    for col in ["bid_cnt", "offer_cnt"]:
        if col not in cnt.columns:
            cnt[col] = 0
    cnt["imbalance"] = (cnt["bid_cnt"] - cnt["offer_cnt"]) / (
        cnt["bid_cnt"] + cnt["offer_cnt"]
    ).replace({0: np.nan})
    return cnt.reset_index()


def atm_otm_mix(df_norm):
    return (
        df_norm.groupby(["underlying", "moneyness_class"])
        .size()
        .reset_index(name="count")
        .sort_values(["underlying", "count"], ascending=[True, False])
    )


# ----------------------------
# Visualization Helper
# ----------------------------
def _tenor_distribution(df_norm):
    df_ten = df_norm.copy()
    if "tenor_days" not in df_ten.columns:
        exp = pd.to_datetime(df_ten.get("expiry"), errors="coerce")
        biz = pd.to_datetime(df_ten.get("business_date"), errors="coerce")
        df_ten["tenor_days"] = (exp - biz).dt.days

    df_ten = df_ten[df_ten["tenor_days"] >= 0].copy()
    df_ten["tenor_bucket"] = df_ten["tenor_days"].apply(_categorize_tenor)

    side_col = [c for c in df_ten.columns if c.lower() == "side"]
    df_ten["side_clean"] = (
        df_ten[side_col[0]].astype(str).str.strip().str.upper() if side_col else ""
    )
    df_ten["is_ioi"] = df_ten["side_clean"].isin(["B", "O"])
    df_ten["is_trade"] = df_ten["side_clean"].eq("T")

    tenor_counts = (
        df_ten.groupby("tenor_bucket")
        .agg(ioi_count=("is_ioi", "sum"), trade_count=("is_trade", "sum"))
        .reset_index()
    )
    tenor_counts = tenor_counts[tenor_counts["tenor_bucket"].notna()].copy()
    tenor_counts["sort_val"] = tenor_counts["tenor_bucket"].apply(_tenor_sort_value)
    tenor_counts = tenor_counts.sort_values("sort_val").reset_index(drop=True)
    return tenor_counts


def build_figures(act, lead, mix, df_norm, imbalance) -> Mapping[str, plt.Figure]:
    figs = {}

    # 1️⃣ Activity by Hour
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(act["hour"], act["ioi_cnt"], label="IOIs", alpha=0.7)
    ax1.bar(act["hour"], act["trade_cnt"], bottom=act["ioi_cnt"], label="Trades", alpha=0.7)
    ax1.set_title("IOI vs Trade Activity by Hour (NY)")
    ax1.set_xlabel("Hour (NY)")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(range(0, 24))
    ax1.set_xticklabels(range(0, 24), rotation=45)
    fig1.tight_layout()
    figs["chart_activity_by_hour"] = fig1

    # 2️⃣ Top Underlyings
    top_leads = lead.nlargest(15, "notional_sum")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(top_leads["underlying"], top_leads["notional_sum"] / 1e6)
    ax2.set_title("Top 15 Underlyings by Notional (MM)")
    ax2.set_xlabel("Notional (Millions)")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    figs["chart_leaders_notional"] = fig2

    # 3️⃣ Moneyness Mix
    mix_tot = mix.groupby("moneyness_class")["count"].sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.pie(
        mix_tot["count"],
        labels=mix_tot["moneyness_class"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax3.set_title("Moneyness Mix")
    fig3.tight_layout()
    figs["chart_moneyness_mix"] = fig3

    # 4️⃣ Tenor Distribution
    tenor_counts = _tenor_distribution(df_norm)
    x = np.arange(len(tenor_counts))
    fig4, (ax_ioi, ax_trd) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_ioi.bar(x, tenor_counts["ioi_count"], color="tab:blue")
    ax_ioi.set_title("Tenor Distribution – IOIs")
    ax_ioi.set_ylabel("IOI Count")
    ax_ioi.grid(alpha=0.3, axis="y")

    ax_trd.bar(x, tenor_counts["trade_count"], color="tab:orange")
    ax_trd.set_title("Tenor Distribution – Trades")
    ax_trd.set_xlabel("Tenor Bucket")
    ax_trd.set_ylabel("Trade Count")
    ax_trd.grid(alpha=0.3, axis="y")

    ax_trd.set_xticks(x)
    ax_trd.set_xticklabels(tenor_counts["tenor_bucket"], rotation=45)
    fig4.suptitle("Tenor Distribution by Type (Rounded & Ordered)", fontsize=13)
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    figs["chart_tenor_distribution"] = fig4

    # 5️⃣ Bid/Offer Imbalance
    if not imbalance.empty:
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.plot(imbalance["date"], imbalance["imbalance"], marker="o", color="tab:blue")
        ax5.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax5.set_title("Daily Bid–Offer Imbalance")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Imbalance")
        ax5.grid(alpha=0.3)
        fig5.autofmt_xdate()
        fig5.tight_layout()
        figs["chart_buy_sell_imbalance"] = fig5

    return figs


def build_excel_report(
    figs: Mapping[str, plt.Figure],
    completeness: pd.DataFrame,
    cov_tenor: pd.DataFrame,
    cov_exp: pd.DataFrame,
    summary_match: pd.DataFrame,
    lifecycle_detail: pd.DataFrame,
    act: pd.DataFrame,
    lead: pd.DataFrame,
    mix: pd.DataFrame,
    report_path: str | None = None,
) -> BytesIO:
    buffer = BytesIO()
    os.makedirs(os.path.dirname(report_path or DEFAULT_REPORT_NAME) or ".", exist_ok=True)
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as xw:
        workbook = xw.book

        for sheet_name, fig in figs.items():
            sheet = workbook.add_worksheet(sheet_name[:31])
            imgdata = BytesIO()
            fig.savefig(imgdata, format="png", bbox_inches="tight", dpi=120)
            sheet.insert_image("B2", f"{sheet_name}.png", {"image_data": imgdata})

        completeness.to_excel(xw, sheet_name="completeness", index=False)
        cov_tenor.to_excel(xw, sheet_name="coverage_tenor", index=False)
        cov_exp.to_excel(xw, sheet_name="coverage_expiry", index=False)
        summary_match.to_excel(xw, sheet_name="ioi_trade_match_summary", index=False)
        lifecycle_detail.to_excel(xw, sheet_name="ioi_trade_lifecycle", index=False)
        act.to_excel(xw, sheet_name="activity_by_hour", index=False)
        lead.to_excel(xw, sheet_name="leaders_underlyings", index=False)
        mix.to_excel(xw, sheet_name="atm_otm_mix", index=False)

    buffer.seek(0)
    return buffer


# ----------------------------
# Streamlit UI
# ----------------------------
@st.cache_data(show_spinner=False)
def _cached_load(ioi_bytes: bytes | None, trd_bytes: bytes | None, use_sample: bool):
    ioi_source = IOI_PATH if use_sample else BytesIO(ioi_bytes) if ioi_bytes else None
    trd_source = TRD_PATH if use_sample else BytesIO(trd_bytes) if trd_bytes else None
    return load_inputs(ioi_source, trd_source)


def _display_metric_row(summary_match, lead):
    total_msgs = int(lead["msgs"].sum()) if not lead.empty else 0
    trade_rate = (
        summary_match["match_rate_pct"].iloc[0] if not summary_match.empty else np.nan
    )
    top_underlying = lead.iloc[0]["underlying"] if not lead.empty else "N/A"

    col1, col2, col3 = st.columns(3)
    col1.metric("Total messages", f"{total_msgs:,}")
    trade_label = f"{trade_rate:0.1f}%" if pd.notna(trade_rate) else "—"
    col2.metric("IOIs with trades", trade_label)
    col3.metric("Top underlying (by notional)", top_underlying)


def main():
    st.set_page_config(
        page_title="Equity Options IOI Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Equity Options IOI + Trade Explorer")
    st.markdown(
        "Upload IOI and trade CSVs or use the default sample files to explore coverage, "
        "lifecycle matching, and tenor distributions. You can also download an Excel "
        "report with the same charts and tables."
    )

    with st.sidebar:
        st.header("Data Inputs")
        use_sample = st.checkbox(
            "Use default sample CSVs", value=True, help="Paths defined in the constants."
        )
        ioi_file = None
        trd_file = None
        if use_sample:
            st.caption(f"IOI: `{IOI_PATH}`")
            st.caption(f"Trades: `{TRD_PATH}`")
        else:
            ioi_file = st.file_uploader("IOI CSV", type=["csv"])
            trd_file = st.file_uploader("Trades CSV", type=["csv"])
            if not ioi_file or not trd_file:
                st.info("Please upload both IOI and trade CSV files to continue.")
        st.caption("If using defaults, files are resolved relative to this app's directory.")
        report_name = st.text_input(
            "Report filename",
            value=DEFAULT_REPORT_NAME,
            help="Name for the downloadable Excel report",
        )

    data_ready = use_sample or (ioi_file and trd_file)
    if not data_ready:
        st.stop()

    try:
        ioi_bytes = ioi_file.getvalue() if ioi_file else None
        trd_bytes = trd_file.getvalue() if trd_file else None
        df_raw, df = _cached_load(ioi_bytes, trd_bytes, use_sample)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data: {exc}")
        if use_sample:
            st.info(
                "Tip: Ensure the sample CSVs exist at "
                f"`{_resolve_path(IOI_PATH)}` and `{_resolve_path(TRD_PATH)}` "
                "or switch off 'Use default sample CSVs' and upload files directly."
            )
        st.stop()

    completeness = field_completeness(df)
    cov_tenor, cov_exp = coverage_tables(df)
    summary_match, lifecycle_detail = ioi_trade_match(df)
    act = activity_by_hour(df)
    lead = leaders(df)
    imbalance = buy_sell_imbalance(df, lead.head())
    mix = atm_otm_mix(df)

    figs = build_figures(act, lead, mix, df, imbalance)

    _display_metric_row(summary_match, lead)

    tabs = st.tabs(
        [
            "Charts",
            "Coverage",
            "Lifecycle",
            "Data quality",
            "Leaders",
            "Imbalance",
            "Download",
        ]
    )

    with tabs[0]:
        st.subheader("Activity by hour")
        st.pyplot(figs["chart_activity_by_hour"])
        st.subheader("Top underlyings by notional")
        st.pyplot(figs["chart_leaders_notional"])
        st.subheader("Moneyness mix")
        st.pyplot(figs["chart_moneyness_mix"])
        st.subheader("Tenor distribution")
        st.pyplot(figs["chart_tenor_distribution"])

    with tabs[1]:
        st.subheader("Coverage by tenor")
        st.dataframe(cov_tenor, use_container_width=True, height=320)
        st.subheader("Coverage by expiry")
        st.dataframe(cov_exp, use_container_width=True, height=320)

    with tabs[2]:
        st.subheader("IOI to trade match summary")
        st.dataframe(summary_match, use_container_width=True)
        st.subheader("Lifecycle detail (per INTEREST_ID)")
        st.dataframe(lifecycle_detail, use_container_width=True, height=400)

    with tabs[3]:
        st.subheader("Field completeness")
        st.dataframe(completeness, use_container_width=True, height=400)

    with tabs[4]:
        st.subheader("Top underlyings")
        st.dataframe(lead, use_container_width=True, height=400)

    with tabs[5]:
        st.subheader("Bid/offer imbalance")
        if imbalance.empty:
            st.info("No bid/offer data available for imbalance calculation.")
        else:
            fig_key = "chart_buy_sell_imbalance"
            if fig_key in figs:
                st.pyplot(figs[fig_key])
            st.dataframe(imbalance, use_container_width=True, height=300)

    with tabs[6]:
        st.subheader("Download Excel report")
        report_buffer = build_excel_report(
            figs,
            completeness,
            cov_tenor,
            cov_exp,
            summary_match,
            lifecycle_detail,
            act,
            lead,
            mix,
            report_path=report_name,
        )
        st.download_button(
            "Download report",
            data=report_buffer,
            file_name=os.path.basename(report_name) or DEFAULT_REPORT_NAME,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Includes all tables plus embedded chart images.")

    st.success(
        f"Processed {df.shape[0]:,} normalized rows (raw combined: {df_raw.shape[0]:,})."
    )


if __name__ == "__main__":
    main()
