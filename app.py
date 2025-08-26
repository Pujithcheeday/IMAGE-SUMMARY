# app.py
import os
import json
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
import streamlit as st
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai

# Optional TTS engine (offline) - used only if available
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# -----------------------
# Config & Bootstrap
# -----------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")
st.set_page_config(page_title="Generative AI Vision App", page_icon="✨", layout="wide")

# CSS + small animations (glass, glow, button bounce, chat bubbles)
st.markdown(
    """
    <style>
    /* page bg & fonts */
    .stApp { background: linear-gradient(180deg, #0f1724 0%, #071022 100%); color: #E6EEF8; }

    /* glowing upload card */
    .upload-card {
        border-radius: 14px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        text-align: center;
        transition: transform .28s ease, box-shadow .28s ease;
    }
    .upload-card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(0,186,255,0.06); }
    .glow { animation: pulse 2.8s ease-in-out infinite; }
    @keyframes pulse {
      0% { box-shadow: 0 0 8px rgba(0,186,255,0.06); }
      50% { box-shadow: 0 0 24px rgba(0,186,255,0.12); }
      100% { box-shadow: 0 0 8px rgba(0,186,255,0.06); }
    }

    /* chat bubble */
    .bubble-user { background: linear-gradient(90deg,#0ea5e9,#7dd3fc); color: #021024; padding:10px 14px; border-radius:12px; width:fit-content; max-width:70%; }
    .bubble-assistant { background: rgba(255,255,255,0.03); color: #E6EEF8; padding:10px 14px; border-radius:12px; width:fit-content; max-width:80%; }

    /* send button */
    .send-btn { border-radius: 10px; padding: 10px 20px; background: linear-gradient(90deg,#ff6b6b,#ff9a9e); color:white; border:none; cursor:pointer; transition: transform .12s ease; }
    .send-btn:active { transform: translateY(2px) scale(.995); }

    /* small utilities */
    .muted { color: #9AA8BF; font-size: 13px; }
    .control-card { border-radius:10px; padding:12px; background: rgba(255,255,255,0.01); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Helper utilities
# -----------------------
def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

HISTORY_FILE = "history.json"

def save_history_to_disk():
    if not st.session_state.get("persist_opt_in", False):
        return
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"saved_at": now_iso(), "items": st.session_state.history}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history to disk: {e}")

def load_history_from_disk():
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("items", [])
            if isinstance(items, list):
                st.session_state.history = items
    except Exception:
        pass

def export_history_bytes():
    payload = {"exported_at": now_iso(), "items": st.session_state.history}
    b = BytesIO()
    b.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    b.seek(0)
    return b

def download_text_bytes(text: str, filename="ai_response.txt"):
    buf = BytesIO()
    buf.write(text.encode("utf-8"))
    buf.seek(0)
    return buf

# Gemini model init (lazy)
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    except Exception:
        model = None
else:
    model = None

def generate_answer(question: str, image_obj: Image.Image):
    if not model:
        raise RuntimeError("Generative model not configured (missing/invalid GOOGLE_API_KEY).")
    # Gemini accepts PIL image in SDK as per earlier usage
    resp = model.generate_content([question, image_obj])
    return resp.text or "(No text returned)"

# -----------------------
# Session state & default
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts {id, timestamp, question, answer, rating, pinned}
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "persist_opt_in" not in st.session_state:
    st.session_state.persist_opt_in = False
if "achievements" not in st.session_state:
    st.session_state.achievements = {"questions_today": 0, "first_upload_done": False}

# If user opted in previously, load
if st.session_state.persist_opt_in:
    load_history_from_disk()

# -----------------------
# Layout: Sidebar Controls
# -----------------------
with st.sidebar:
    st.header("⚙️ App Controls")
    st.caption("Privacy-first: images kept in memory only; text is optional persisted.")
    st.session_state.persist_opt_in = st.checkbox("📥 Persist history to disk (text-only)", value=st.session_state.persist_opt_in, help="When enabled, conversation text will be stored in history.json in this folder.")
    if st.button("🗑️ Clear Session History"):
        st.session_state.history = []
        save_history_to_disk()
        st.success("Session history cleared.")
    st.download_button("⬇️ Export History (JSON)", data=export_history_bytes(), file_name="vision_history.json", mime="application/json")
    st.markdown("---")
    st.subheader("📝 Conversation History")
    if st.session_state.history:
        # show most recent 6 with quick actions
        for entry in reversed(st.session_state.history[-6:]):
            ts = entry.get("timestamp","")
            q = entry.get("question","")
            a = entry.get("answer","")
            pinned = entry.get("pinned", False)
            star = "⭐" if entry.get("rating", 0) >= 4 else ""
            st.markdown(f"**{star} {q}**  \n_{ts}_  {'📌' if pinned else ''}")
            st.write(a if len(a) < 240 else a[:240]+"…")
            cols = st.columns([0.18,0.82])
            if cols[0].button("📌", key=f"pin_{ts}"):
                entry["pinned"] = not pinned
                save_history_to_disk()
                st.experimental_rerun()
    else:
        st.info("No history yet — upload an image and ask a question to begin.")

# -----------------------
# Main area
# -----------------------
st.title("IMAGE SUMMARY")
st.caption("User-friendly • Privacy-first • Emoji + animation-rich UX")

# Info / pro tip
st.markdown(
    "<div class='control-card'>"
    "<strong class='muted'>Pro tip:</strong> Use presets for quick insights, or write a custom question. Toggle persistence only if you want local text logs.</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2], gap="large")

# ---------- Left Column: Upload + Presets ----------
with col1:
    st.subheader("📤 Upload Image")
    st.markdown("<div class='upload-card glow'>Drag & drop or browse • JPG/PNG • up to 200MB</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], accept_multiple_files=False, key="uploader")
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.session_state.current_image = img
            st.image(img, caption="Preview", use_column_width=True)
            if not st.session_state.achievements["first_upload_done"]:
                st.session_state.achievements["first_upload_done"] = True
        except UnidentifiedImageError:
            st.error("Could not decode image. Upload a valid JPG/PNG.")
            st.session_state.current_image = None
    else:
        st.info("Upload an image to interact with the AI.")

    st.markdown("---")
    st.subheader("🎯 Presets")
    presets = {
        "Summarize this image": "Summarize this image in 2-3 sentences, include the main subjects and mood.",
        "List key objects in the scene": "List all visible objects and approximate counts.",
        "Describe colors, lighting, and mood": "Describe the colors, lighting conditions and overall mood.",
        "Explain likely context & purpose": "Explain what might be happening and the likely context of the scene.",
        "Identify potential safety concerns": "List any potential safety hazards or issues visible in the scene.",
        "Suggest social media caption (short)": "Write a short, catchy social media caption with 1-2 emojis."
    }
    preset_choice = st.selectbox("Quick-pick prompt", list(presets.keys()))
    preset_text = presets[preset_choice]

# ---------- Right Column: Chat / Q&A ----------
with col2:
    st.subheader("💬 Ask the AI about this image")
    # show conversation (last 10)
    if st.session_state.history:
        for item in st.session_state.history[-10:]:
            st.markdown(f"<div class='bubble-user'>{item['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bubble-assistant'>{item['answer']}</div>", unsafe_allow_html=True)

    prompt = st.text_input("Your question", value=preset_text, key="prompt_input", placeholder="Try: 'Summarize this image'")

    # small row for buttons and toggles
    cols = st.columns([0.18,0.18,0.18,0.46])
    send_clicked = cols[0].button("🚀 Send", key="send_btn")
    play_tts = cols[1].button("🔊 Play last (TTS)" if TTS_AVAILABLE else "🔇 TTS N/A", key="tts_btn")
    cols[2].download_button("💾 Download last", data=download_text_bytes(st.session_state.history[-1]["answer"]) if st.session_state.history else "", file_name="latest_answer.txt")
    cols[3].checkbox("I understand text-only persistence", value=True, key="ack_persist")

    # quick-mode buttons
    st.markdown("<div style='margin-top:6px'>Quick actions: ", unsafe_allow_html=True)
    qa_cols = st.columns([1,1,1,1])
    if qa_cols[0].button("📝 Summarize"):
        prompt = presets["Summarize this image"]
        st.session_state.prompt_input = prompt
    if qa_cols[1].button("🔎 Detect objects"):
        prompt = "List all visible objects and their probable labels."
        st.session_state.prompt_input = prompt
    if qa_cols[2].button("😂 Make it funny"):
        prompt = "Describe this image in a humorous, light-hearted way with emojis."
        st.session_state.prompt_input = prompt
    if qa_cols[3].button("📸 Caption"):
        prompt = "Suggest 3 short social-media captions (with emojis)."
        st.session_state.prompt_input = prompt
    st.markdown("</div>", unsafe_allow_html=True)

    # Send logic
    if send_clicked:
        if not API_KEY:
            st.error("GOOGLE_API_KEY not configured. Add it to your environment or .env file.")
        elif st.session_state.current_image is None:
            st.error("Please upload an image first.")
        elif not prompt.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("🤖 Generating response..."):
                try:
                    answer = generate_answer(prompt.strip(), st.session_state.current_image)
                except Exception as e:
                    answer = f"Error: {e}"
            # record
            entry = {
                "id": f"{datetime.now().timestamp()}",
                "timestamp": now_iso(),
                "question": prompt.strip(),
                "answer": answer,
                "rating": 0,
                "pinned": False
            }
            st.session_state.history.append(entry)
            st.session_state.achievements["questions_today"] += 1
            save_history_to_disk()
            # render new chat bubble
            st.markdown(f"<div class='bubble-user'>{entry['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bubble-assistant'>{entry['answer']}</div>", unsafe_allow_html=True)

    # TTS play for last answer (best-effort)
    if play_tts:
        if not TTS_AVAILABLE:
            st.warning("TTS engine not available. Install pyttsx3 to enable offline TTS.")
        elif not st.session_state.history:
            st.info("No answers available yet to play.")
        else:
            last_ans = st.session_state.history[-1]["answer"]
            try:
                engine = pyttsx3.init()
                engine.say(last_ans)
                engine.runAndWait()
            except Exception as e:
                st.warning(f"TTS playback failed: {e}")

    # Per-message actions area (rating, pin, download)
    if st.session_state.history:
        last = st.session_state.history[-1]
        rr = st.slider("Rate the last answer (1-5)", min_value=1, max_value=5, value=3, key="rate_slider")
        if st.button("⭐ Save Rating"):
            last["rating"] = rr
            save_history_to_disk()
            st.success("Thanks for rating!")
        if st.button("📌 Pin last"):
            last["pinned"] = not last.get("pinned", False)
            save_history_to_disk()
            st.info("Pinned toggled.")

# -----------------------
# Bottom: Achievements & Footer
# -----------------------
st.markdown("---")
ach = st.session_state.achievements
st.markdown(f"**🔥 Questions asked this session:** {ach['questions_today']}  •  **First upload done:** {'✅' if ach['first_upload_done'] else '❌'}")

st.caption("No images are persisted. Text persistence is opt-in only. © Vision App")

# small JS helper for clipboard copy (works on the page)
st.markdown(
    """
    <script>
    const copyBtns = document.querySelectorAll('[data-copy-target]')
    function copyText(t) {
      navigator.clipboard.writeText(t).then(()=>{console.log("copied")}, ()=>{console.log("failed")})
    }
    </script>
    """,
    unsafe_allow_html=True,
)
