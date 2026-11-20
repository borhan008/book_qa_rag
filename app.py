import streamlit as st
import requests


sidebar = st.sidebar.title("Sidebar")
st.sidebar.write("This is the sidebar where you can add additional controls and information.")

st.markdown("""
    <style>

    /* Pin the form to the bottom */
    [data-testid="stForm"] {
          /* background-color: #f0f2f6; */
    position: fixed;
    /* left: 0; */
    /* right: 0; */
    bottom: 20px;
    z-index: 99999999;
    height: auto;
    border-top: 1px solid #333;
    padding-top: 25px;
    width: 50%;
    }
    </style>
""", unsafe_allow_html=True)



if "messages" not in st.session_state:
    st.session_state.messages = []

if "typing" not in st.session_state:
    st.session_state.typing = False

top_section = st.container()

with top_section:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=":material/account_circle:" if msg["role"] == "user" else ":material/assistant:"):
            st.write(msg["content"])


if st.session_state.typing:
    with st.chat_message("assistant", avatar=":material/assistant:"):
        st.write("Assistant is thinking..")
else:
    st.session_state.typing = False


with st.form("my_form", clear_on_submit=True, border=False):
    col1, col2 = st.columns([10, 1], vertical_alignment="center")
    with col1:
        text = st.text_input("Enter some text:", key="text_input", required=True, label_visibility="collapsed", placeholder="Write your question here...")
    with col2:
        submitted  = st.form_submit_button("", icon=":material/arrow_forward:", key="submit_button",  use_container_width=True)


if submitted and text.strip():
    st.session_state.messages.append({"role": "user", "content": text})

    with st.spinner("উত্তর খোঁজা হচ্ছে..."):
        try:
            response = requests.get(
                "http://localhost:8000/ask",
                params={"question": text},
                timeout=120,
            )
            response.raise_for_status()
            answer = response.json()["response"]
        except requests.exceptions.HTTPError:
            answer = f"সার্ভারে সমস্যা হয়েছে: {response.json().get('detail', response.text)}"
        except requests.exceptions.RequestException as e:
            answer = f"সার্ভারের সাথে সংযোগ করা যায়নি: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

