import pandas as pd

from analytics import summarize_hospitalization_data


def test_hospitalization_summary():
    data = pd.DataFrame(
        {
            "Region": ["East", "West"],
            "Hospitalizations per 10,000": [40, 60],
            "Population": [1_000_000, 1_200_000],
        }
    )

    result = summarize_hospitalization_data(data)

    assert result["average_rate"] == 50.0
    assert result["highest_rate_region"] == "West"
    assert result["regions_analyzed"] == 2
