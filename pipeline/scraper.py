import pandas as pd

def fetch_upi_data():

    url = "https://www.npci.org.in/what-we-do/upi/product-statistics"

    tables = pd.read_html(url)

    df = tables[0]

    return df