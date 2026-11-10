import streamlit as st
import pandas as pd
import json
import html
from datetime import datetime

from complaint_database import (
    create_database,
    get_all_complaints,
    save_complaint,
    update_complaint_status
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
    padding: 21px;
    min-height: 140px;
    box-shadow: 0 5px 18px rgba(32,59,120,0.035);
}

.metric-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.metric-label {
    color: #8290A5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.metric-icon {
    width: 37px;
    height: 37px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    font-size: 17px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: #203B78;
    margin-top: 13px;
    line-height: 1.2;
}

.metric-note {
    font-size: 11px;
    color: #9AA5B5;
    margin-top: 7px;
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
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">{label}</div>
            <div class="metric-icon" style="background:{icon_bg}">
                {icon}
            </div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
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
            "Submit Complaint",
            "Complaint Management",
            "Similarity Analysis",
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
# DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="page-heading">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Your central overview of student complaints, priorities and resolutions.'
        '</div>',
        unsafe_allow_html=True
    )

    # METRICS

    total = len(df)

    pending = len(df[df["status"] == "Pending"]) if total else 0
    progress = len(df[df["status"] == "In Progress"]) if total else 0
    resolved = len(df[df["status"] == "Resolved"]) if total else 0

    high_priority = (
        len(df[df["priority"].isin(["High", "Critical"])])
        if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL COMPLAINTS",
            total,
            "📋",
            "#E8EDF8",
            "All recorded submissions"
        )

    with c2:
        metric_card(
            "PENDING",
            pending,
            "🕒",
            "#FFF4D6",
            "Awaiting attention"
        )

    with c3:
        metric_card(
            "IN PROGRESS",
            progress,
            "⚡",
            "#E6EEFF",
            "Currently being addressed"
        )

    with c4:
        metric_card(
            "RESOLVED",
            resolved,
            "✓",
            "#E2F4E9",
            "Successfully completed"
        )

    st.write("")
    st.write("")

    # ANALYTICS

    st.markdown(
        '<div class="section-heading">Complaint Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Understand where complaints originate and how urgent they are.'
        '</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        chart1, chart2 = st.columns(2)

        with chart1:

            with st.container(border=True):

                st.markdown("**Department Distribution**")

                department_counts = (
                    df["department"]
                    .value_counts()
                    .rename_axis("Department")
                    .to_frame("Complaints")
                )

                st.bar_chart(
                    department_counts,
                    color="#4164AD",
                    horizontal=True
                )

        with chart2:

            with st.container(border=True):

                st.markdown("**Urgency Breakdown**")

                urgency_order = ["High", "Medium", "Low"]

                urgency_counts = (
                    df["urgency"]
                    .value_counts()
                    .reindex(urgency_order, fill_value=0)
                    .rename_axis("Urgency")
                    .to_frame("Complaints")
                )

                st.bar_chart(
                    urgency_counts,
                    color="#E8C96B"
                )

        st.write("")
        st.write("")

        # PRIORITY OVERVIEW

        left, right = st.columns([1, 1])

        with left:

            with st.container(border=True):

                st.markdown("**Priority Overview**")

                priority_counts = (
                    df["priority"]
                    .value_counts()
                    .reindex(
                        ["Critical", "High", "Medium", "Low"],
                        fill_value=0
                    )
                )

                for priority, count in priority_counts.items():

                    percentage = (
                        count / total * 100 if total else 0
                    )

                    st.markdown(
                        f"**{priority}**"
                        f"<span style='float:right;color:#8290A5'>"
                        f"{count}</span>",
                        unsafe_allow_html=True
                    )

                    st.progress(percentage / 100)

        with right:

            with st.container(border=True):

                st.markdown("**Resolution Progress**")

                resolution_rate = (
                    resolved / total * 100 if total else 0
                )

                st.markdown(
                    f"<div style='font-size:35px;font-weight:800;"
                    f"color:#203B78'>{resolution_rate:.1f}%</div>",
                    unsafe_allow_html=True
                )

                st.caption("Complaints marked as resolved")

                st.progress(resolution_rate / 100)

                st.caption(
                    f"{resolved} resolved out of {total} total complaints"
                )

        st.write("")
        st.write("")

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

        recent = df.head(5)

        for _, row in recent.iterrows():

            complaint_text = html.escape(
                str(row["complaint_text"])
            )

            department = html.escape(str(row["department"]))
            status = str(row["status"])
            priority = str(row["priority"])

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
                <div class="activity-meta">{department}</div>
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
# SUBMIT COMPLAINT
# =====================================================

elif page == "Submit Complaint":

    st.markdown(
        '<div class="page-heading">Submit Complaint</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Describe your campus issue and let CampusPulse analyse it.</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown("### Complaint Details")

        with st.form("complaint_form"):

            complaint_text = st.text_area(
                "Describe your complaint",
                placeholder="Describe your campus issue in detail...",
                height=160,
                max_chars=2000
            )

            submitted = st.form_submit_button(
                "Analyse & Submit Complaint",
                type="primary",
                use_container_width=True
            )

    if submitted:

        if len(complaint_text.strip()) < 10:

            st.warning("Please enter a more detailed complaint.")

        else:

            try:

                with st.spinner("Analysing your complaint..."):

                    from nlp_pipeline import process_complaint

                    result = process_complaint(complaint_text.strip())

                    complaint_id = save_complaint(
                        complaint_text.strip(),
                        result
                    )

                    st.session_state["last_submission"] = {
                        "id": complaint_id,
                        "text": complaint_text.strip(),
                        "result": result
                    }

                st.success(
                    f"Complaint CP-{complaint_id:04d} submitted successfully!"
                )

            except Exception as error:

                st.error("Unable to process your complaint.")
                st.exception(error)

    submission = st.session_state.get("last_submission")

    if submission:

        result = submission["result"]

        st.divider()

        st.markdown("### Complaint Analysis")

        st.caption(f"Complaint ID: CP-{submission['id']:04d}")

        st.markdown("**Submitted Complaint**")
        st.write(submission["text"])

        departments = result["department_predictions"]
        urgency = result["urgency"]
        sentiment = result["sentiment"]
        priority = result["priority"]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "DEPARTMENT",
                departments[0]["department"] if departments else "Unknown",
                "🏢", "#E8EDF8", "Predicted department"
            )

        with c2:
            metric_card(
                "URGENCY",
                urgency["urgency"],
                "⚡", "#FFF4D6", "Predicted urgency"
            )

        with c3:
            metric_card(
                "SENTIMENT",
                sentiment["sentiment"],
                "💬", "#E6EEFF", "Detected sentiment"
            )

        with c4:
            metric_card(
                "PRIORITY",
                priority["priority"],
                "🚩", "#FCE8E7", "Calculated priority"
            )

        if priority["needs_human_review"]:
            st.warning("This complaint requires human review.")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Department & Urgency",
            "Sentiment & Priority",
            "Keywords & Entities",
            "Similar Complaints"
        ])

        with tab1:

            st.markdown("#### Department Predictions")

            if departments:

                department_df = pd.DataFrame(departments)

                department_df["Model Score (%)"] = (
                    department_df["confidence"].astype(float)
                )

                department_df.loc[
                    department_df["Model Score (%)"] <= 1,
                    "Model Score (%)"
                ] *= 100

                st.dataframe(
                    department_df.rename(
                        columns={"department": "Department"}
                    )[["Department", "Model Score (%)"]],
                    use_container_width=True,
                    hide_index=True
                )

            st.caption("Model scores are not guaranteed probabilities.")

            st.markdown("#### Urgency Analysis")

            urgency_df = pd.DataFrame(
                list(urgency["probabilities"].items()),
                columns=["Urgency", "Score"]
            )

            urgency_df["Score"] = urgency_df["Score"].astype(float)

            urgency_df.loc[
                urgency_df["Score"] <= 1,
                "Score"
            ] *= 100

            st.bar_chart(
                urgency_df.set_index("Urgency"),
                color="#4164AD"
            )

        with tab2:

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Sentiment")
                st.metric("Detected Sentiment", sentiment["sentiment"])
                st.metric(
                    "Model Score",
                    f"{float(sentiment['confidence']):.2f}%"
                )

            with c2:
                st.markdown("#### Priority")
                st.metric("Priority Level", priority["priority"])
                st.metric(
                    "Priority Score",
                    f"{float(priority['score']):.2f}/100"
                )

            st.write(
                "Critical indicator:",
                "Yes" if priority["critical_indicator"] else "No"
            )

            st.write(
                "Time-sensitive:",
                "Yes" if priority["time_sensitive"] else "No"
            )

            st.write(
                "Human review required:",
                "Yes" if priority["needs_human_review"] else "No"
            )

        with tab3:

            st.markdown("#### Extracted Keywords")

            keywords = result.get("keywords", [])

            if keywords:
                st.write(" · ".join(keywords))
            else:
                st.info("No keywords identified.")

            st.divider()

            st.markdown("#### Named Entities")

            entities = result.get("entities", [])

            if entities:
                st.dataframe(
                    pd.DataFrame(entities),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No entities identified.")

        with tab4:

            st.markdown("#### Sentence-BERT Matches")

            for index, match in enumerate(
                result.get("semantic_similar_complaints", []), 1
            ):

                with st.container(border=True):

                    st.markdown(
                        f"**Match {index} — {match['similarity']:.2f}% similarity**"
                    )

                    st.write(match["complaint"])

                    st.caption(
                        f"{match['department']} | {match['urgency']}"
                    )

            st.divider()

            st.markdown("#### TF-IDF Matches")

            for index, match in enumerate(
                result.get("tfidf_similar_complaints", []), 1
            ):

                with st.container(border=True):

                    st.markdown(
                        f"**Match {index} — {match['similarity']:.2f}% similarity**"
                    )

                    st.write(match["complaint"])

                    st.caption(
                        f"{match['department']} | {match['urgency']}"
                    )


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
# SIMILARITY ANALYSIS
# =====================================================

elif page == "Similarity Analysis":

    st.markdown(
        '<div class="page-heading">Similarity Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Find complaints describing similar campus issues.</div>',
        unsafe_allow_html=True
    )

    query = st.text_area(
        "Enter a complaint",
        placeholder="Example: The Wi-Fi in the hostel keeps disconnecting.",
        height=130
    )

    if st.button(
        "Find Similar Complaints",
        type="primary",
        use_container_width=True
    ):

        if len(query.strip()) < 10:

            st.warning("Please enter a more detailed complaint.")

        else:

            try:

                with st.spinner("Comparing complaints..."):

                    from nlp_pipeline import (
                        find_similar_complaints,
                        find_tfidf_similarity
                    )

                    semantic = find_similar_complaints(query.strip())
                    tfidf = find_tfidf_similarity(query.strip())

                    st.session_state["similarity_results"] = {
                        "semantic": semantic,
                        "tfidf": tfidf
                    }

            except Exception as error:

                st.error("Similarity analysis failed.")
                st.exception(error)

    results = st.session_state.get("similarity_results")

    if results:

        tab1, tab2 = st.tabs(["Sentence-BERT", "TF-IDF"])

        with tab1:

            st.markdown("### Semantic Similarity")

            for index, match in enumerate(results["semantic"], 1):

                with st.container(border=True):

                    st.markdown(f"**Match {index}**")
                    st.write(match["complaint"])

                    score = min(
                        max(float(match["similarity"]) / 100, 0),
                        1
                    )

                    st.progress(score)

                    st.caption(
                        f"{match['similarity']:.2f}% similarity | "
                        f"{match['department']} | {match['urgency']}"
                    )

        with tab2:

            st.markdown("### TF-IDF Similarity")

            for index, match in enumerate(results["tfidf"], 1):

                with st.container(border=True):

                    st.markdown(f"**Match {index}**")
                    st.write(match["complaint"])

                    score = min(
                        max(float(match["similarity"]) / 100, 0),
                        1
                    )

                    st.progress(score)

                    st.caption(
                        f"{match['similarity']:.2f}% similarity | "
                        f"{match['department']} | {match['urgency']}"
                    )


# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":

    st.markdown(
        '<div class="page-heading">Reports & Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Explore complaint statistics and download reports.</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.info("Reports will appear once complaints are submitted.")

    else:

        total = len(df)

        resolved = (
            df["status"].eq("Resolved").sum()
        )

        resolution_rate = (
            resolved / total * 100 if total else 0
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "TOTAL COMPLAINTS",
                total,
                "📋",
                "#E8EDF8",
                "All recorded submissions"
            )

        with c2:
            metric_card(
                "RESOLVED",
                resolved,
                "✓",
                "#E2F4E9",
                "Successfully completed"
            )

        with c3:
            metric_card(
                "RESOLUTION RATE",
                f"{resolution_rate:.1f}%",
                "📈",
                "#FFF4D6",
                "Resolved complaints"
            )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.markdown("### Department Report")

            department_report = (
                df["department"]
                .value_counts()
                .rename_axis("Department")
                .to_frame("Complaints")
            )

            st.bar_chart(
                department_report,
                color="#4164AD",
                horizontal=True
            )

        with right:

            st.markdown("### Urgency Report")

            urgency_report = (
                df["urgency"]
                .value_counts()
                .rename_axis("Urgency")
                .to_frame("Complaints")
            )

            st.bar_chart(
                urgency_report,
                color="#E8C96B"
            )

        left, right = st.columns(2)

        with left:

            st.markdown("### Status Report")

            status_report = (
                df["status"]
                .value_counts()
                .rename_axis("Status")
                .to_frame("Complaints")
            )

            st.bar_chart(status_report)

        with right:

            st.markdown("### Priority Report")

            priority_report = (
                df["priority"]
                .value_counts()
                .rename_axis("Priority")
                .to_frame("Complaints")
            )

            st.bar_chart(priority_report)

        st.divider()

        st.markdown("### Download Reports")

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
            "Download Complaint Report (CSV)",
            data=csv,
            file_name="campuspulse_report.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    CAMPUSPULSE &nbsp; • &nbsp; SMARTER COMPLAINT MANAGEMENT
</div>
""", unsafe_allow_html=True)
