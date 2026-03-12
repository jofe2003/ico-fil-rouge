import pandas as pd
df = pd.read_excel(r"C:\Users\natyj\Documents\ICO\Program\ico-fil-rouge\CommonFiles\6_detail_table_cust_depots_distances.xls")
df.to_json("6_detail_table_cust_depots_distances.json", orient="records", indent=4)