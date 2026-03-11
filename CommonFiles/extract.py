import panda as pd
df = pd.read_excel("2_detail_table_customers.xlsx")
df.to_json("2_detail_table_customers.json", orient="records", indent=4)