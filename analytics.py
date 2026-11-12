def summarize_hospitalization_data(data):
    """
    Calculate summary statistics for regional hospitalization data.
    """

    required_columns = [
        "Region",
        "Hospitalizations per 10,000",
        "Population",
    ]

    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    rates = data["Hospitalizations per 10,000"]

    return {
        "average_rate": rates.mean(),
        "highest_rate_region": data.loc[rates.idxmax(), "Region"],
        "regions_analyzed": len(data),
    }
