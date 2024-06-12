import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, message, from_email, to_email, password):
    email = MIMEMultipart()
    email['From'] = from_email
    email['To'] = to_email
    email['Subject'] = subject
    email.attach(MIMEText(message, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    server.login(from_email, password)

    server.sendmail(from_email, to_email, email.as_string())

    server.quit()

# Usage
subject = "Test Email"
message = "This is a test email sent from Python."
from_email = "your.email@gmail.com"
to_email = "recipient.email@gmail.com"
password = "your_password"

send_email(subject, message, from_email, to_email, password)
