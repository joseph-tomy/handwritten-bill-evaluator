import tempfile

import streamlit as st

from production import ProductionPipeline

st.set_page_config(
    page_title="Handwritten Bill Evaluator",
    page_icon="🧾",
)

st.title("🧾 Handwritten Bill Evaluator")

st.success("Winning Model : Gemini")

pipeline = ProductionPipeline()

uploaded = st.file_uploader(
    "Upload Bill",
    type=["jpg", "jpeg", "png"],
)

if uploaded:

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg",
    )

    temp.write(uploaded.read())
    temp.close()

    with st.spinner("Extracting bill..."):

        prediction = pipeline.extract(temp.name)

    st.subheader("Extracted Information")

    vendor = st.text_input(
        "Vendor",
        prediction.get("vendor"),
    )

    bill_no = st.text_input(
        "Bill Number",
        prediction.get("bill_no"),
    )

    date = st.text_input(
        "Date",
        prediction.get("date"),
    )

    amount = st.text_input(
        "Amount",
        prediction.get("amount"),
    )

    currency = st.text_input(
        "Currency",
        prediction.get("currency"),
    )

    gst = st.text_input(
        "GST",
        prediction.get("gst"),
    )

    if st.button("Create Expense"):

        prediction = {
            "vendor": vendor,
            "bill_no": bill_no,
            "date": date,
            "amount": amount,
            "currency": currency,
            "gst": gst,
        }

        try:

            expense = pipeline.create_expense(prediction)

            st.success("Expense Created Successfully")

            st.json(expense)

        except Exception as e:

            st.error(str(e))