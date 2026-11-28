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
        sender_email = st.secrets["sender_email"]
        app_password = st.secrets["app_password"]
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
