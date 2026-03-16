import pyodbc

def get_connection():

    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=.;"
        "DATABASE=NPCI_Data;"
        "Trusted_Connection=yes;"
    )

    return conn