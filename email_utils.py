import smtplib
from email.message import EmailMessage
import streamlit as st

def send_email(recipient: str, subject: str, body: str) -> bool:
    """Send a plain‑text email via Gmail SMTP.

    Parameters
    ----------
    recipient: str
        Email address of the recipient.
    subject: str
        Subject line of the email.
    body: str
        Plain‑text body of the email.

    Returns
    -------
    bool
        ``True`` if the message was sent successfully, ``False`` otherwise.
    """
    try:
        # Load credentials from Streamlit secrets – never hard‑code them.
        sender_email = st.secrets["general"]["sender_email"]
        app_password = st.secrets["general"]["app_password"]
    except Exception as e:
        print(f"[email_utils] Error reading credentials from st.secrets: {e}")
        return False

    # Build the email message.
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        # Connect to Gmail's SMTP server with STARTTLS.
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True
    except Exception as e:
        # Gracefully handle any SMTP errors.
        print(f"[email_utils] Failed to send email to {recipient}: {e}")
        return False


def send_complaint_notification(complaint_id, complaint_text, result):
    from department_routing import get_department_email

    try:
        departments = result.get("department_predictions", [])
        if not departments:
            return False

        department = departments[0].get("department", "Unknown")
        recipient = get_department_email(department)

        if not recipient:
            print(f"[email_utils] No recipient configured for {department}")
            return False

        urgency_data = result.get("urgency", {})
        priority_data = result.get("priority", {})

        urgency = urgency_data.get("urgency", "Unknown")
        priority = priority_data.get("priority", "Unknown")
        score = priority_data.get("score", "N/A")

        subject = f"CampusPulse | New Complaint CP-{complaint_id:04d}"

        body = f"""
A new campus complaint has been submitted.

Complaint ID: CP-{complaint_id:04d}
Department: {department}
Urgency: {urgency}
Priority: {priority}
Priority Score: {score}

Complaint:
{complaint_text}

Please review this complaint through the CampusPulse dashboard.
"""

        return send_email(recipient, subject, body)

    except Exception as e:
        print(f"[email_utils] Complaint notification error: {e}")
        return False