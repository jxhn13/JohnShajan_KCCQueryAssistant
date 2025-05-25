import streamlit as st
from backend.load_kcc import generate_answer, load_kcc_data


st.set_page_config(page_title="🌾 KCC Query Assistant", layout="centered")

if "kcc_loaded" not in st.session_state:
    with st.spinner("🔄 Loading KCC dataset..."):
        load_kcc_data(force_reload=True)
        st.session_state["kcc_loaded"] = True
    st.success("✅ KCC data ready!")


st.markdown("<h1 style='text-align: center; color: green;'>🌾 KCC Query Assistant</h1>", unsafe_allow_html=True)


query = st.text_input("💬 Ask your agriculture-related question:", value=st.session_state.get("query_input", ""), key="query_input")


if st.button("Ask"):
    if query.strip():
        with st.spinner("🔍 Thinking..."):
            result = generate_answer(query.strip())  

        local = result.get("local_answer")
        internet = result.get("internet_answer")

        st.markdown("#### 🧠 Structured Answer")

        if local:
            with st.expander("📗 Advice from KCC Database", expanded=True):
                st.markdown(
                    f"""
                    <div style='background-color: #f0fdf4;
                                padding: 16px;
                                border-radius: 12px;
                                border: 1px solid #bbf7d0;
                                color: black;
                                font-size: 16px;
                                line-height: 1.6;
                                font-family: "Segoe UI", sans-serif;'>
                        {local}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if internet:
            with st.expander("🛰️ Fallback Answer from Live Internet Search", expanded=not local):
                st.markdown(
                    f"""
                    <div style='background-color: #fff7ed;
                                padding: 16px;
                                border-radius: 12px;
                                border: 1px solid #fdba74;
                                color: #78350f;
                                font-size: 16px;
                                line-height: 1.6;
                                font-family: "Segoe UI", sans-serif;'>
                        {internet}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if not local and not internet:
            st.warning("⚠️ Sorry, we couldn't find a suitable answer from either local KCC data or the internet.")
