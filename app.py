import streamlit as st
import pandas as pd
import json
import html
from datetime import datetime
import re

from complaint_database import (
    add_complaint_support,
    create_database,
    get_all_complaints,
    save_complaint,
    update_complaint_status,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CampusPulse | Admin Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
create_database()

# =====================================================
# THEME AND CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --navy: #203B78;
    --blue: #4164AD;
    --yellow: #E8C96B;
    --cream: #F5F3E9;
    --text: #26334D;
    --muted: #8290A5;
}

.stApp {
    background: var(--cream);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: none;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D6DEF0;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.16);
}

[data-testid="stSidebar"] .stRadio label {
    padding: 8px 10px;
    border-radius: 8px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.10);
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--navy) !important;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1px;
    color: white;
}

.brand-dot {
    color: var(--yellow);
}

.sidebar-caption {
    font-size: 12px;
    color: #C7D2E8;
    margin-top: 3px;
    letter-spacing: 0.5px;
}

.sidebar-section {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    color: #AFC0E2;
    margin: 25px 0 10px 5px;
}

.page-heading {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -1.2px;
    margin-bottom: 2px;
}

.page-subtitle {
    color: #8290A5;
    font-size: 14px;
    margin-bottom: 25px;
}

.top-tag {
    display: inline-block;
    background: #E8EDF8;
    color: #4164AD;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E7EAF0;
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 130px;
    height: auto;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 5px 18px rgba(32,59,120,0.035);
    box-sizing: border-box;
    overflow: hidden;
}

.metric-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.metric-label {
    color: #8290A5;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.metric-icon {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    font-size: 15px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #203B78;
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
    display: block;
}

.metric-note {
    font-size: 11px;
    color: #9AA5B5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.section-heading {
    font-family: 'Manrope', sans-serif;
    color: #203B78;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 3px;
}

.section-description {
    color: #8995A7;
    font-size: 12px;
    margin-bottom: 15px;
}

.panel {
    background: white;
    border: 1px solid #E7EAF0;
    border-radius: 16px;
    padding: 21px;
    box-shadow: 0 5px 18px rgba(32,59,120,0.035);
}

.activity-row {
    background: white;
    border: 1px solid #E7EAF0;
    border-radius: 12px;
    padding: 14px 17px;
    margin-bottom: 9px;
}

.activity-id {
    color: #4164AD;
    font-weight: 800;
    font-size: 12px;
}

.activity-text {
    color: #26334D;
    font-size: 13px;
    font-weight: 600;
    margin-top: 5px;
}

.activity-meta {
    color: #929DAE;
    font-size: 11px;
    margin-top: 5px;
}

.status-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 5px 10px;
    font-size: 10px;
    font-weight: 800;
    white-space: nowrap;
}

.status-pending {
    background: #FFF4D6;
    color: #966A13;
}

.status-progress {
    background: #E6EEFF;
    color: #4164AD;
}

.status-resolved {
    background: #E2F4E9;
    color: #28734A;
}

.priority-high {
    background: #FCE8E7;
    color: #B43F39;
}

.priority-medium {
    background: #FFF3D8;
    color: #966A13;
}

.priority-low {
    background: #E7F1EA;
    color: #39744F;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E7EAF0;
    border-radius: 12px;
    overflow: hidden;
}

.stSelectbox > div > div,
.stTextInput > div > div {
    border-radius: 9px;
    background: white;
}

div.stButton > button {
    background: #4164AD;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 9px 17px;
}

div.stButton > button:hover {
    background: #203B78;
    color: white;
    border: none;
}

.empty-state {
    background: white;
    border: 1px dashed #CCD5E4;
    border-radius: 16px;
    padding: 45px 20px;
    text-align: center;
    color: #8290A5;
}

.empty-icon {
    font-size: 35px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #A0A9B7;
    font-size: 11px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def safe_json(value, default=None):

    if default is None:
        default = []

    if isinstance(value, (list, dict)):
        return value

    if value is None:
        return default

    try:
        if pd.isna(value) or value == "":
            return default
    except (TypeError, ValueError):
        pass

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default

def get_department(value):
    predictions = safe_json(value)

    if predictions and isinstance(predictions, list):
        return predictions[0].get("department", "Unassigned")

    return "Unassigned"


def metric_card(label, value, icon, icon_bg, note):
    val_str = str(value)
    if len(val_str) > 10:
        font_size = "15px"
    elif len(val_str) > 7:
        font_size = "18px"
    elif len(val_str) > 4:
        font_size = "22px"
    else:
        font_size = "26px"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">{label}</div>
            <div class="metric-icon" style="background:{icon_bg}">
                {icon}
            </div>
        </div>
        <div class="metric-value" style="font-size: {font_size};" title="{val_str}">{value}</div>
        <div class="metric-note" title="{note}">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# LOAD DATABASE
# =====================================================

try:
    complaints = get_all_complaints()
    df = pd.DataFrame(complaints)
except Exception as error:
    st.error(f"Could not load the complaint database: {error}")
    st.stop()


if not df.empty:
    df["department"] = df["department_predictions"].apply(get_department)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Campus<span class="brand-dot">Pulse.</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">STUDENT COMPLAINT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">WORKSPACE</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Submit & Analyze",
            "Complaint Management",
            "Reports"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-section">SYSTEM</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-caption">NLP ENGINE ACTIVE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">DATABASE CONNECTED</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-caption">CampusPulse • Version 1.0</div>',
        unsafe_allow_html=True
    )


# =====================================================
# PAGE-CHANGE STATE CLEANUP
# Clear submission state when navigating away from Submit & Analyze
# so the form always starts clean on a fresh visit.
# =====================================================

_prev_page = st.session_state.get("_active_page", page)
if _prev_page != page:
    st.session_state.pop("last_submission", None)
    st.session_state.pop("pending_submission", None)
    st.session_state.pop("just_supported_id", None)
st.session_state["_active_page"] = page


# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="page-heading">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'A quick status overview and the latest student complaints.'
        '</div>',
        unsafe_allow_html=True
    )

    # METRICS

    report_counts = (
        1 + df["support_count"].fillna(0).astype(int)
        if not df.empty
        else pd.Series(dtype="int64")
    )
    total = int(report_counts.sum())

    if not df.empty:
        pending = int(report_counts[df["status"] == "Pending"].sum())
        progress = int(report_counts[df["status"] == "In Progress"].sum())
        resolved = int(report_counts[df["status"] == "Resolved"].sum())
    else:
        pending = progress = resolved = 0

    dash_c1, dash_c2, dash_c3, dash_c4 = st.columns(4)

    with dash_c1:
        metric_card(
            "TOTAL COMPLAINTS",
            total,
            "📋",
            "#E8EDF8",
            "All recorded submissions"
        )

    with dash_c2:
        metric_card(
            "PENDING",
            pending,
            "🕒",
            "#FFF4D6",
            "Awaiting attention"
        )

    with dash_c3:
        metric_card(
            "IN PROGRESS",
            progress,
            "⚡",
            "#E6EEFF",
            "Currently being addressed"
        )

    with dash_c4:
        metric_card(
            "RESOLVED",
            resolved,
            "✓",
            "#E2F4E9",
            "Successfully completed"
        )

    st.write("")
    st.write("")

    if not df.empty:

        # RECENT COMPLAINTS

        st.markdown(
            '<div class="section-heading">Recent Complaints</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'The latest complaints submitted to CampusPulse.'
            '</div>',
            unsafe_allow_html=True
        )

        recent = df.head(10)

        for _, row in recent.iterrows():

            complaint_text = html.escape(
                str(row["complaint_text"])
            )

            department = html.escape(str(row["department"]))
            status = str(row["status"])
            priority = str(row["priority"])
            support_count = int(row.get("support_count", 0) or 0)
            support_note = (
                f" · {support_count} additional +1 report(s)"
                if support_count else ""
            )

            status_class = {
                "Pending": "status-pending",
                "In Progress": "status-progress",
                "Resolved": "status-resolved"
            }.get(status, "status-pending")

            priority_class = {
                "Critical": "priority-high",
                "High": "priority-high",
                "Medium": "priority-medium",
                "Low": "priority-low"
            }.get(priority, "priority-low")

            st.markdown(f"""
            <div class="activity-row">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;gap:10px;flex-wrap:wrap">
                    <span class="activity-id">COMPLAINT #{row['id']}</span>
                    <div>
                        <span class="status-pill {priority_class}">
                            {html.escape(priority)} PRIORITY
                        </span>
                        <span class="status-pill {status_class}">
                            {html.escape(status)}
                        </span>
                    </div>
                </div>
                <div class="activity-text">{complaint_text}</div>
                <div class="activity-meta">{department}{support_note}</div>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📋</div>
            <h3>No complaints yet</h3>
            <p>New student submissions will appear here.</p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# PLACEHOLDER PAGES
# =====================================================

# =====================================================
# =====================================================
# SUBMIT & ANALYZE COMPLAINT
# =====================================================

elif page in ["Submit & Analyze", "Submit Complaint"]:

    st.markdown(
        '<div class="page-heading">Submit & Analyze Complaint</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Report a campus issue, check for recurring duplicates (+1), and view instant AI intelligence classification.</div>',
        unsafe_allow_html=True
    )

    just_supported_id = st.session_state.pop("just_supported_id", None)
    if just_supported_id:
        st.success(
            f"✅ Endorsement recorded! Added your +1 to Complaint CP-{just_supported_id:04d}."
        )

    with st.container(border=True):

        st.markdown("### Complaint Submission")

        with st.form("complaint_form"):

            complaint_text = st.text_area(
                "Describe your complaint",
                placeholder="Describe your campus issue in detail (e.g., Wi-Fi keeps disconnecting in Block C, or water leakage in Hostel 2nd floor)...",
                height=140,
                max_chars=2000
            )

            submitted = st.form_submit_button(
                "Check & Submit Complaint",
                type="primary",
                use_container_width=True
            )

    if submitted:

        if len(complaint_text.strip()) < 10:

            st.warning("Please enter a more detailed complaint (at least 10 characters).")

        else:

            try:

                with st.spinner("Running AI analysis & checking for existing duplicates..."):

                    from nlp_pipeline import (
                        find_similar_submitted_complaints,
                        process_complaint,
                    )

                    result = process_complaint(complaint_text.strip())
                    result["submitted_complaint_matches"] = (
                        find_similar_submitted_complaints(
                            complaint_text.strip(),
                            complaints,
                        )
                    )

                    st.session_state["pending_submission"] = {
                        "text": complaint_text.strip(),
                        "result": result
                    }

            except Exception as error:

                st.error("Unable to process your complaint.")
                st.exception(error)

    pending_submission = st.session_state.get("pending_submission")

    if pending_submission:
        pending_matches = pending_submission["result"].get(
            "submitted_complaint_matches", []
        )

        if pending_matches:
            st.warning(
                "🔍 **Similar complaints already exist in the system!**\n\n"
                "If one of the issues below matches yours, you can click **'+1 This Issue'** to endorse it instead of filing a duplicate. "
                "Your endorsement is included in the recurring issue and report totals."
            )

            for match in pending_matches:
                with st.container(border=True):
                    col_info, col_btn = st.columns([0.72, 0.28])
                    with col_info:
                        st.markdown(
                            f"**Complaint CP-{match['id']:04d} — "
                            f"{match['similarity']:.1f}% similar**"
                        )
                        st.write(match["complaint"])
                        st.caption(f"Current Status: **{match['status']}**")
                    with col_btn:
                        st.write("")
                        if st.button(
                            "👍 +1 This Issue",
                            key=f"support_match_{match['id']}",
                            type="primary",
                            use_container_width=True,
                            help="Endorse this existing complaint instead of filing a duplicate ticket."
                        ):
                            try:
                                add_complaint_support(int(match["id"]))
                                st.session_state["just_supported_id"] = match["id"]
                                del st.session_state["pending_submission"]
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not add support: {e}")

            st.write("")
            st.markdown("##### Is your issue different or newly occurring?")
            submit_col, cancel_col = st.columns(2)
            if submit_col.button(
                "Submit as a new complaint anyway",
                type="secondary",
                key="submit_after_duplicate_check",
                use_container_width=True
            ):
                complaint_id = save_complaint(
                    pending_submission["text"],
                    pending_submission["result"],
                )
                st.session_state["last_submission"] = {
                    "id": complaint_id,
                    **pending_submission,
                }
                del st.session_state["pending_submission"]
                st.success(
                    f"Complaint CP-{complaint_id:04d} submitted successfully!"
                )
                st.rerun()

            if cancel_col.button(
                "Cancel submission",
                key="cancel_after_duplicate_check",
                use_container_width=True
            ):
                del st.session_state["pending_submission"]
                st.rerun()
        else:
            complaint_id = save_complaint(
                pending_submission["text"],
                pending_submission["result"],
            )
            st.session_state["last_submission"] = {
                "id": complaint_id,
                **pending_submission,
            }
            del st.session_state["pending_submission"]
            st.success(
                f"Complaint CP-{complaint_id:04d} submitted successfully!"
            )
            st.rerun()

    submission = (
        None
        if pending_submission
        else st.session_state.get("last_submission")
    )

    if submission:

        result = submission["result"]

        st.divider()

        col_title, col_reset = st.columns([0.75, 0.25])
        with col_title:
            st.markdown(f"### AI Intelligence Analysis — Ticket CP-{submission['id']:04d}")
        with col_reset:
            st.write("")
            if st.button("🔄 Submit Another", key="reset_submission", use_container_width=True):
                del st.session_state["last_submission"]
                st.rerun()

        with st.container(border=True):
            st.markdown("**Submitted Complaint Text:**")
            st.write(submission["text"])

        departments = result["department_predictions"]
        urgency = result["urgency"]
        sentiment = result["sentiment"]
        emotion = result.get("emotion", {"emotion": "Neutral", "cues": []})
        priority = result["priority"]

        # 4 CORE AI METRIC CARDS
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            dept_name = departments[0]["department"] if departments else "Unknown"
            dept_conf = departments[0]["confidence"] if departments else 0
            metric_card(
                "DEPARTMENT",
                dept_name,
                "🏢",
                "#E8EDF8",
                f"Confidence: {float(dept_conf):.0f}%"
            )

        with c2:
            metric_card(
                "URGENCY",
                urgency["urgency"],
                "⚡",
                "#FFF4D6",
                f"Score: {float(urgency.get('confidence', 0)):.0f}%"
            )

        with c3:
            sent_label = sentiment.get("sentiment", "Neutral")
            sent_conf = sentiment.get("confidence", 0)
            metric_card(
                "SENTIMENT",
                sent_label,
                "💬",
                "#FCE8E7" if sent_label.lower() == "negative" else "#E2F4E9",
                f"Tone Score: {float(sent_conf):.0f}%"
            )

        with c4:
            prio_label = priority.get("priority", "Medium")
            prio_score = priority.get("score", 0)
            metric_card(
                "PRIORITY",
                prio_label,
                "🚩",
                "#FCE8E7" if prio_label in ["Critical", "High"] else "#FFF4D6",
                f"Score: {float(prio_score):.0f}/100"
            )

        if priority.get("needs_human_review"):
            st.warning("⚠️ This complaint requires administrative human review.")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Department & Urgency",
            "Sentiment & Priority",
            "Keywords & Entities",
            "Similar Complaints"
        ])

        with tab1:
            col_d, col_u = st.columns(2)
            with col_d:
                st.markdown("#### Department Predictions")
                if departments:
                    dept_df = pd.DataFrame(departments)
                    dept_df["Model Score (%)"] = dept_df["confidence"].astype(float)
                    dept_df.loc[dept_df["Model Score (%)"] <= 1, "Model Score (%)"] *= 100
                    st.dataframe(
                        dept_df.rename(columns={"department": "Department"})[["Department", "Model Score (%)"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption("Model scores are not guaranteed probabilities.")
                else:
                    st.info("No department predictions available.")
            with col_u:
                st.markdown("#### Urgency Breakdown")
                if "probabilities" in urgency:
                    urg_df = pd.DataFrame(
                        list(urgency["probabilities"].items()),
                        columns=["Urgency Level", "Probability (%)"]
                    )
                    urg_df["Probability (%)"] = urg_df["Probability (%)"].astype(float)
                    urg_df.loc[urg_df["Probability (%)"] <= 1, "Probability (%)"] *= 100
                    st.dataframe(
                        urg_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Urgency breakdown not available.")

        with tab2:
            sent_label = sentiment.get("sentiment", "Neutral")
            sent_conf = sentiment.get("confidence", 0)
            prio_label = priority.get("priority", "Medium")
            prio_score = priority.get("score", 0)

            col_s, col_e, col_p = st.columns(3)
            with col_s:
                st.markdown("#### Sentiment Analysis")
                sentiment_color = "#FCE8E7" if sent_label.lower() == "negative" else "#E2F4E9"
                st.markdown(
                    f'<div style="background:{sentiment_color};border-radius:10px;padding:14px 18px;">'
                    f'<div style="font-size:13px;color:#8290A5;font-weight:700;">DETECTED TONE</div>'
                    f'<div style="font-size:28px;font-weight:800;color:#203B78;">{sent_label}</div>'
                    f'<div style="font-size:12px;color:#9AA5B5;">Confidence: {float(sent_conf):.0f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_e:
                st.markdown("#### Emotion Cues")
                st.metric("Detected Emotion", emotion["emotion"])
                if emotion["cues"]:
                    st.caption(
                        "Matched wording: "
                        + ", ".join(f"`{cue}`" for cue in emotion["cues"])
                    )
                else:
                    st.caption("No clear emotion wording detected.")
            with col_p:
                st.markdown("#### Priority Score")
                st.markdown(
                    f'<div style="background:#FFF4D6;border-radius:10px;padding:14px 18px;">'
                    f'<div style="font-size:13px;color:#8290A5;font-weight:700;">CALCULATED PRIORITY</div>'
                    f'<div style="font-size:28px;font-weight:800;color:#203B78;">{prio_label}</div>'
                    f'<div style="font-size:12px;color:#9AA5B5;">Score: {float(prio_score):.0f}/100</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with tab3:
            k_col, e_col = st.columns(2)
            with k_col:
                st.markdown("#### Extracted Keywords")
                keywords = result.get("keywords", [])
                if keywords:
                    st.write(" · ".join([f"`{k}`" for k in keywords]))
                else:
                    st.info("No distinct keywords identified.")

            with e_col:
                st.markdown("#### Identified Entities & Locations")
                entities = result.get("entities", [])
                if entities:
                    st.dataframe(
                        pd.DataFrame(entities),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No specific entities or campus locations identified.")

        with tab4:
            st.markdown("#### Semantic Reference Matches (Sentence-BERT)")
            sem_matches = result.get("semantic_similar_complaints", [])
            if sem_matches:
                for index, match in enumerate(sem_matches[:3], 1):
                    with st.container(border=True):
                        st.markdown(f"**Match #{index}** — {match['similarity']:.1f}% similarity")
                        st.write(match["complaint"])
                        st.caption(f"Department: **{match['department']}** · Urgency: **{match['urgency']}**")
            else:
                st.info("No similar past complaints found in the training reference set.")

    # BROWSE ALL OPEN CAMPUS ISSUES
    st.write("")
    st.divider()
    with st.expander("🔍 Browse Open Recurring Issues on Campus (+1 Endorsements)"):
        st.caption("Active campus complaints grouped by recurring topics. You can add a +1 to any problem you are also experiencing.")
        try:
            from nlp_pipeline import find_recurring_submitted_complaint_groups
            recurring_groups = find_recurring_submitted_complaint_groups(complaints)
            if recurring_groups:
                for group in recurring_groups:
                    with st.container(border=True):
                        st.markdown(f"**{group['topic_label']}** — *{group['count']} report(s)*")
                        representative = group["complaints"][0]
                        st.write(representative["complaint"])
                        st.caption(
                            f"Status: **{representative['status']}** · "
                            f"{representative['support_count']} additional +1 report(s)"
                        )
                        if st.button(
                            "👍 +1 I'm affected too",
                            key=f"rec_support_{representative['id']}"
                        ):
                            add_complaint_support(int(representative["id"]))
                            st.session_state["just_supported_id"] = representative["id"]
                            st.rerun()
            else:
                st.info("No open recurring issue clusters found.")
        except Exception as e:
            st.caption(f"Recurring clustering unavailable: {e}")


# =====================================================
# COMPLAINT MANAGEMENT
# =====================================================

elif page == "Complaint Management":

    st.markdown(
        '<div class="page-heading">Complaint Management</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Search, filter and manage student complaints.</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.info("No complaints available.")

    else:

        search = st.text_input(
            "Search complaints",
            placeholder="Search complaint text or ID..."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            departments = ["All"] + sorted(
                df["department"].dropna().unique().tolist()
            )
            selected_department = st.selectbox(
                "Department", departments
            )

        with c2:
            priorities = ["All"] + sorted(
                df["priority"].dropna().unique().tolist()
            )
            selected_priority = st.selectbox(
                "Priority", priorities
            )

        with c3:
            statuses = ["All", "Pending", "In Progress", "Resolved"]
            selected_status = st.selectbox(
                "Status", statuses
            )

        filtered = df.copy()

        if search:
            filtered = filtered[
                filtered["complaint_text"].fillna("").str.contains(
                    search, case=False, na=False
                )
                |
                filtered["id"].astype(str).str.contains(
                    search, case=False, na=False
                )
            ]

        if selected_department != "All":
            filtered = filtered[
                filtered["department"] == selected_department
            ]

        if selected_priority != "All":
            filtered = filtered[
                filtered["priority"] == selected_priority
            ]

        if selected_status != "All":
            filtered = filtered[
                filtered["status"] == selected_status
            ]

        st.caption(
            f"Showing {len(filtered)} of {len(df)} complaints"
        )

        for _, complaint in filtered.iterrows():

            with st.expander(
                f"Complaint #{complaint['id']} | "
                f"{complaint['department']} | "
                f"{complaint['priority']}"
            ):

                st.write(complaint["complaint_text"])

                a, b, c = st.columns(3)

                a.write(f"**Department:** {complaint['department']}")
                b.write(f"**Urgency:** {complaint['urgency']}")
                c.write(f"**Priority:** {complaint['priority']}")

                st.write(
                    f"**Submitted:** {complaint['submitted_at']}"
                )
                support_count = int(
                    complaint.get("support_count", 0) or 0
                )
                if support_count:
                    st.write(
                        f"**Additional +1 reports:** {support_count}"
                    )

                current_status = complaint.get("status", "Pending")

                options = ["Pending", "In Progress", "Resolved"]

                if current_status not in options:
                    current_status = "Pending"

                new_status = st.selectbox(
                    "Update status",
                    options,
                    index=options.index(current_status),
                    key=f"status_{complaint['id']}"
                )

                if st.button(
                    "Save Status",
                    key=f"save_{complaint['id']}"
                ):

                    updated = update_complaint_status(
                        int(complaint["id"]),
                        new_status
                    )

                    if updated:
                        st.success("Status updated successfully.")
                        st.rerun()
                    else:
                        st.error("Could not update the complaint.")





# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":

    st.markdown(
        '<div class="page-heading">Reports & Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Department performance audit, resolution tracking, and executive briefing.</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.info("Reports will appear once complaints are submitted.")

    else:

        report_counts = 1 + df["support_count"].fillna(0).astype(int)
        total = int(report_counts.sum())
        pending = int(report_counts[df["status"].eq("Pending")].sum())
        in_progress = int(report_counts[df["status"].eq("In Progress")].sum())
        resolved = int(report_counts[df["status"].eq("Resolved")].sum())
        resolution_rate = (resolved / total * 100) if total else 0.0

        high_prio_mask = df["priority"].isin(["Critical", "High"]) & df["status"].ne("Resolved")
        critical_backlog = int(report_counts[high_prio_mask].sum())
        active_departments = df["department"].nunique()

        # 1. KPI CARDS (CLEAN & PRACTICAL)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "TOTAL COMPLAINTS",
                total,
                "📋",
                "#E8EDF8",
                f"{len(df)} distinct submissions"
            )

        with c2:
            metric_card(
                "RESOLUTION RATE",
                f"{resolution_rate:.0f}%",
                "📈",
                "#E2F4E9" if resolution_rate >= 50 else "#FFF4D6",
                f"{resolved} of {total} resolved"
            )

        with c3:
            metric_card(
                "PENDING BACKLOG",
                pending,
                "⏳",
                "#FCE8E7" if pending > 0 else "#E2F4E9",
                f"{critical_backlog} high priority" if critical_backlog > 0 else "0 high priority"
            )

        with c4:
            metric_card(
                "DEPARTMENTS",
                active_departments,
                "🏢",
                "#E8EDF8",
                "Units with active reports"
            )

        st.divider()

        # 2. EXECUTIVE BRIEFING BOX
        st.markdown("### 📋 Executive Summary")

        department_totals = (
            df.assign(report_count=report_counts)
            .groupby("department")["report_count"]
            .sum()
            .sort_values(ascending=False)
        )
        top_dept = department_totals.index[0] if not department_totals.empty else "N/A"
        top_dept_count = int(department_totals.iloc[0]) if not department_totals.empty else 0
        top_dept_pct = (top_dept_count / total * 100) if total else 0

        known_locations = [
            "Block A", "Block B", "Block C", "Block D", "Block E", "Hostel",
            "Girls Hostel", "Boys Hostel", "Library", "Cafeteria", "Mess",
            "Auditorium", "Science Lab", "Computer Lab", "Parking Area", "Admin Block"
        ]
        location_counts = {}
        for text, report_count in zip(
            df["complaint_text"].fillna(""),
            report_counts,
        ):
            for loc in known_locations:
                if re.search(r"\b" + re.escape(loc) + r"\b", text, re.IGNORECASE):
                    location_counts[loc] = (
                        location_counts.get(loc, 0) + int(report_count)
                    )

        top_location = max(location_counts, key=location_counts.get) if location_counts else None
        top_location_hits = location_counts.get(top_location, 0) if top_location else 0

        crit_pending_depts = [str(d) for d in df[high_prio_mask]["department"].unique() if pd.notna(d)]
        crit_pending_str = ", ".join(crit_pending_depts) if crit_pending_depts else "None"

        with st.container(border=True):
            b1, b2 = st.columns(2)
            with b1:
                if critical_backlog > 0:
                    st.markdown(f"🚨 **Urgent Backlog:** **{critical_backlog}** high-priority complaints pending in **{crit_pending_str}**.")
                else:
                    st.markdown("✅ **Priority Status:** No critical or high-priority complaints are currently pending.")

                st.markdown(f"🏢 **Top Department:** **{top_dept}** represents **{top_dept_pct:.0f}%** of total volume ({top_dept_count} complaints).")

            with b2:
                if top_location:
                    st.markdown(f"📍 **Infrastructure Hotspot:** **{top_location}** ({top_location_hits} incident reports recorded).")
                else:
                    st.markdown("📍 **Campus Distribution:** Reports currently span general services with no facility cluster.")

                st.markdown(f"⚡ **Throughput:** Campus-wide resolution sits at **{resolution_rate:.0f}%** ({resolved} closed, {pending} pending).")

        st.divider()

        # 3. DEPARTMENT PERFORMANCE TABLE
        st.markdown("### 🏢 Department Performance")

        scorecard_rows = []
        for dept_name, group in df.groupby("department"):
            g_counts = 1 + group["support_count"].fillna(0).astype(int)
            g_total = int(g_counts.sum())
            g_resolved = int(g_counts[group["status"].eq("Resolved")].sum())
            g_pending = int(g_counts[group["status"].eq("Pending")].sum())
            g_in_prog = int(g_counts[group["status"].eq("In Progress")].sum())
            g_crit = int(g_counts[group["priority"].isin(["Critical", "High"])].sum())
            g_rate = (g_resolved / g_total * 100) if g_total else 0.0

            if g_rate >= 70:
                health = "🟢 On Track"
            elif g_rate >= 40:
                health = "🟡 Needs Attention"
            else:
                health = "🔴 Backlog" if g_pending > 0 else "🟠 In Progress"

            scorecard_rows.append({
                "Department": dept_name,
                "Total Complaints": g_total,
                "Critical / High": g_crit,
                "Pending": g_pending,
                "In Progress": g_in_prog,
                "Resolved": g_resolved,
                "Resolution Rate": f"{g_rate:.0f}%",
                "Status": health
            })

        scorecard_df = pd.DataFrame(scorecard_rows).sort_values("Total Complaints", ascending=False)

        st.dataframe(
            scorecard_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # 4. VISUAL BREAKDOWN (PROPORTIONAL & COMPACT)
        col_left, col_right = st.columns(2)

        with col_left:
            with st.container(border=True):
                st.markdown("**Resolution by Department**")
                chart_data = scorecard_df.set_index("Department")[["Resolved", "Pending", "In Progress"]]
                st.bar_chart(
                    chart_data,
                    color=["#28734A", "#B43F39", "#4164AD"],
                    height=200
                )

        with col_right:
            with st.container(border=True):
                st.markdown("**Reported Emotions**")
                if "emotion" in df.columns:
                    emotion_counts = (
                        df.assign(report_count=report_counts)
                        .dropna(subset=["emotion"])
                        .groupby("emotion")["report_count"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                else:
                    emotion_counts = pd.Series(dtype="int64")

                if not emotion_counts.empty:
                    st.bar_chart(
                        emotion_counts,
                        color="#E8C96B",
                        horizontal=True,
                        height=200
                    )
                else:
                    st.info(
                        "Emotion results will appear as new complaints are analyzed."
                    )

        st.divider()

        # 5. EXPORT & EXECUTIVE BRIEFING
        st.markdown("### 📥 Reports & Export")

        current_date = datetime.now().strftime("%B %d, %Y")
        executive_memo = f"""================================================================================
CAMPUSPULSE: EXECUTIVE COMPLAINT INTELLIGENCE REPORT
Date: {current_date}
================================================================================

1. OVERVIEW
--------------------------------------------------------------------------------
- Total Recorded Complaints: {total}
- Campus Resolution Rate: {resolution_rate:.0f}% ({resolved} resolved, {pending} pending, {in_progress} in progress)
- High-Priority Backlog: {critical_backlog} complaint(s)
- Primary Department by Volume: {top_dept} ({top_dept_count} complaints, {top_dept_pct:.0f}%)

2. DEPARTMENT AUDIT
--------------------------------------------------------------------------------
"""
        for _, row in scorecard_df.iterrows():
            executive_memo += (
                f"- {row['Department']:<22} | Total: {row['Total Complaints']:<3} | "
                f"Resolved: {row['Resolved']:<3} ({row['Resolution Rate']:<5}) | "
                f"Pending: {row['Pending']:<3} | Status: {row['Status']}\n"
            )

        executive_memo += f"""
3. RECOMMENDED ACTIONS
--------------------------------------------------------------------------------
- Address the {pending} pending issue(s), prioritizing any critical maintenance or safety items.
- Ensure {top_dept} has sufficient staff allocation to maintain timely turnaround.

================================================================================
CONFIDENTIAL - CAMPUSPULSE GOVERNANCE REPORT
================================================================================
"""

        exp1, exp2 = st.columns(2)

        with exp1:
            st.download_button(
                "📄 Download Executive Briefing (.txt)",
                data=executive_memo,
                file_name=f"CampusPulse_Executive_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )

        with exp2:
            export_df = df.drop(
                columns=[
                    "department_predictions",
                    "keywords",
                    "entities",
                    "semantic_matches",
                    "tfidf_matches"
                ],
                errors="ignore"
            )
            csv = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📊 Download Complaint Data (CSV)",
                data=csv,
                file_name="campuspulse_complaints.csv",
                mime="text/csv",
                use_container_width=True
            )

        with st.expander("Preview Executive Briefing Memo"):
            st.text(executive_memo)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    CAMPUSPULSE &nbsp; • &nbsp; SMARTER COMPLAINT MANAGEMENT
</div>
""", unsafe_allow_html=True)
