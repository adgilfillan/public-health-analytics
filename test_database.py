
import pandas as pd
from sqlalchemy import create_engine, text

# Connect to your local PostgreSQL database
engine = create_engine(
    "postgresql+psycopg:///health_analytics"
)

# Retrieve the research projects
with engine.connect() as connection:
    query = text("SELECT * FROM research_projects")
    results = pd.read_sql(query, connection)

print("\nDatabase connection successful!")
print("\nResearch projects:")
print(results.to_string(index=False))
