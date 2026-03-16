import pandas as pd

def clean_app_names(df):

    # remove spaces
    df["App"] = df["App"].str.strip().str.lower()

    # normalize names
    app_map = {
        "phone pe": "PhonePe",
        "phonepe": "PhonePe",
        "phonepe ": "PhonePe",

        "google pay": "Google Pay",
        "google pay ": "Google Pay",

        "paytm ": "Paytm",
        "paytmwallet": "Paytm",
        "paytm (ocl)": "Paytm",

        "amazon pay ": "Amazon Pay",
        "cred ": "Cred",
        "whatsapp ": "WhatsApp",
        "navi ": "Navi",

        "axis bank ": "Axis Bank",
        "bhim ": "BHIM",
        "mobikwik ": "Mobikwik"
    }

    df["App"] = df["App"].replace(app_map)

    # capitalize remaining
    df["App"] = df["App"].str.title()

    return df