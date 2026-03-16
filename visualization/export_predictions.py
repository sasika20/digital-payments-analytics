def export_predictions(df,path):

    df.to_csv(path,index=False)