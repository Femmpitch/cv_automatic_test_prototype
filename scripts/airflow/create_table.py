from sqlalchemy import create_engine
import pandas as pd

string = "postgresql+psycopg2://superset:superset@postgres_superset:5432/superset"
engine = create_engine(string)

df = pd.read_sql("create table benchmarks_results (name text, result float); select * from benchmarks_results", engine)




