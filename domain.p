# ✅ AI-Powered Personal Data Leak Tracker with Twilio Alerts (Google Colab Compatible)

# -----------------------------
# 📦 STEP 1: Install Required Libraries
# -----------------------------
!pip install pandas numpy scikit-learn gradio matplotlib seaborn plotly transformers spacy reportlab twilio --quiet
!python -m spacy download en_core_web_sm

# -----------------------------
# 📁 STEP 2: Upload Breach Dataset
# -----------------------------
from google.colab import files
import pandas as pd

uploaded = files.upload()  # Upload data_leaks.csv manually

# Load the dataset
df = pd.read_csv("data_leaks.csv")
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# 🧠 STEP 3: Threat Level Classification
# -----------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Assign threat level based on info_type
def assign_threat(row):
    if 'full_profile' in row['info_type']:
        return 'High'
    elif 'pwd' in row['info_type']:
        return 'Medium'
    else:
        return 'Low'

df['threat_level'] = df.apply(assign_threat, axis=1)
le = LabelEncoder()
df['label'] = le.fit_transform(df['threat_level'])

X = pd.get_dummies(df[['info_type']])
y = df['label']

clf = RandomForestClassifier()
clf.fit(X, y)

# -----------------------------
# 🔔 STEP 4: Twilio Real-Time Alert Setup
# -----------------------------
from twilio.rest import Client

# Replace these with your real Twilio credentials
TWILIO_ACCOUNT_SID = 'AC9be8ea651078164b7a547c94df319c3b'
TWILIO_AUTH_TOKEN = '39e82df6b45e6bdc103640c7d1ebac98'
TWILIO_PHONE_NUMBER = '+19787758909'
ALERT_RECEIVER_PHONE = '+919123144609'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_alert_sms(message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=ALERT_RECEIVER_PHONE
    )

# -----------------------------
# 🧪 STEP 5: Leak Checker Logic
# -----------------------------
def check_email_leak(email):
    result = df[df['email'] == email]
    if result.empty:
        return "✅ No breach found."
    else:
        X_test = pd.get_dummies(result[['info_type']])
        X_test = X_test.reindex(columns=X.columns, fill_value=0)
        pred = clf.predict(X_test)
        level = le.inverse_transform(pred)[0]

        message = f"⚠️ Data breach detected for {email}!\nThreat Level: {level}\nSources: {result['source'].values.tolist()}"
        send_alert_sms(message)
        return message

# -----------------------------
# 🎛️ STEP 6: Gradio UI
# -----------------------------
import gradio as gr

def leak_checker_ui(email):
    return check_email_leak(email)

gr.Interface(fn=leak_checker_ui, inputs="text", outputs="text", title="🛡️ AI Leak Checker with SMS Alerts").launch()

# -----------------------------
# 🧾 STEP 7: PDF Report Generator (Optional)
# -----------------------------
from reportlab.pdfgen import canvas

def generate_report(email, report_text):
    file = f"{email}_report.pdf"
    c = canvas.Canvas(file)
    c.drawString(100, 750, f"Leak Report for: {email}")
    c.drawString(100, 730, report_text)
    c.save()
    return file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data
num_rows = 100

emails = [f'user{i}@example.com' for i in range(num_rows)]
passwords = [f'pass{i}word' if i % 2 == 0 else '' for i in range(num_rows)]  # some empty passwords
sources = ['linkedin.com', 'adobe.com', 'dropbox.com', 'twitter.com', 'facebook.com', 'github.com']
info_types = ['email+pwd', 'email']

np.random.seed(42)
df_sample = pd.DataFrame({
    'email': emails,
    'password': passwords,
    'source': np.random.choice(sources, size=num_rows),
    'info_type': [info_types[0] if pwd != '' else info_types[1] for pwd in passwords],
    'date': [datetime.now() - timedelta(days=np.random.randint(0, 1825)) for _ in range(num_rows)]  # last 5 years
})

# Save CSV
df_sample.to_csv('sample_data_leaks.csv', index=False)
# 1. Install required packages
!pip install gradio twilio scikit-learn --quiet

# 2. Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from twilio.rest import Client
import gradio as gr

# 3. Upload dataset (upload your CSV here)
from google.colab import files
uploaded = files.upload()

# 4. Load dataset
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# 5. Data preprocessing
df['date'] = pd.to_datetime(df['date'])

# Create 'threat_level' column based on simple rules
sensitive_sources = ['linkedin.com', 'adobe.com', 'dropbox.com']
df['threat_level'] = df.apply(
    lambda row: 'High' if ('pwd' in row['info_type']) and (row['source'] in sensitive_sources)
    else ('Medium' if 'pwd' in row['info_type'] else 'Low'),
    axis=1
)

# Encode categorical variables
le_source = LabelEncoder()
df['source_enc'] = le_source.fit_transform(df['source'])
le_info = LabelEncoder()
df['info_enc'] = le_info.fit_transform(df['info_type'])
le_threat = LabelEncoder()
df['threat_enc'] = le_threat.fit_transform(df['threat_level'])

# Feature engineering: days since breach
df['days_since_breach'] = (datetime.now() - df['date']).dt.days

# Features and label
X = df[['source_enc', 'info_enc', 'days_since_breach']]
y = df['threat_enc']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accu
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_threat.classes_))

# 8. Twilio config (replace with your actual Twilio SID, Auth Token, and phone numbers)
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = '+1234567890'
ALERT_PHONE_NUMBER = '+0987654321'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_alert(phone, message):
    message = client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=phone
    )
    return message.sid

# 9. Gradio interface for user input and prediction
def predict_leak(email, source, info_type, breach_date):
    try:
        breach_date = pd.to_datetime(breach_date)
    except:
        return "Invalid date format. Use YYYY-MM-DD."

    # Prepare input features
    if source not in le_source.classes_:
        return f"Source '{source}' not recognized. Choose from: {list(le_source.classes_)}"
    if info_type not in le_info.classes_:
        return f"Info Type '{info_type}' not recognized. Choose from: {list(le_info.classes_)}"

    source_enc = le_source.transform([source])[0]
    info_enc = le_info.transform([info_type])[0]
    days_since_breach = (datetime.now() - breach_date).days

    X_input = np.array([[source_enc, info_enc, days_since_breach]])
    pred_enc = clf.predict(X_input)[0]
    pred_label = le_threat.inverse_transform([pred_enc])[0]

    # If threat is high, send alert SMS
    if pred_label == 'High':
        msg = f"Alert! High threat leak detected for email: {email}, source: {source}."
        try:
            sid = send_alert(ALERT_PHONE_NUMBER, msg)
            alert_msg = f"SMS Alert sent with SID: {sid}"
        except Exception as e:
            alert_msg = f"Failed to send SMS alert: {e}"
    else:
        alert_msg = "No alert sent."

    return f"Predicted Threat Level: {pred_label}\n{alert_msg}"

# 10. Setup Gradio UI
source_choices = list(le_source.classes_)
info_type_choices = list(le_info.classes_)

iface = gr.Interface(
    fn=predict_leak,
    inputs=[
        gr.Textbox(label="Email"),
        gr.Dropdown(source_choices, label="Source"),
        gr.Dropdown(info_type_choices, label="Info Type"),
        gr.Textbox(label="Breach Date (YYYY-MM-DD)")
    ],
    outputs="text",
    title="AI Personal Data Leak Threat Predictor",
    description="Enter data leak info to predict threat level and get SMS alert for high threats."
)

iface.launch()
