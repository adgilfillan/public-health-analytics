import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import patsy
from patsy import build_design_matrices


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

data_path = "data/raw/analytic_supplement_20260325[1].csv"

df = pd.read_csv(data_path)

county = df[
    (df["statecode"] > 0) &
    (df["countycode"] > 0)
].copy()

analysis = pd.DataFrame({
    "statecode": county["statecode"],
    "hospital_stays": county["v005_rawvalue"],
    "primary_care_access": county["v004_rawalternatevalue"],
    "obesity": county["v011_rawvalue"],
    "diabetes": county["v060_rawvalue"],
    "uninsured": county["v003_rawvalue"],
    "unemployment": county["v023_rawvalue"]
})

analysis = analysis[
    (analysis["hospital_stays"] > 0) &
    (analysis["primary_care_access"] > 0)
].dropna().copy()

analysis["log_hospital_stays"] = np.log(
    analysis["hospital_stays"]
)

analysis["log_primary_care_access"] = np.log(
    analysis["primary_care_access"]
)

# Center continuous predictors.
for variable in [
    "obesity",
    "diabetes",
    "uninsured",
    "unemployment"
]:
    analysis[f"{variable}_c"] = (
        analysis[variable] - analysis[variable].mean()
    )


# ============================================================
# 2. FIT ADJUSTED QUADRATIC REGRESSION MODEL
# ============================================================

formula = """
log_hospital_stays ~
log_primary_care_access +
obesity_c + I(obesity_c ** 2) +
diabetes_c + I(diabetes_c ** 2) +
uninsured_c +
unemployment_c +
C(statecode)
"""

y, X = patsy.dmatrices(
    formula,
    data=analysis,
    return_type="dataframe"
)

design_info = X.design_info

model = smf.ols(
    formula,
    data=analysis
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": analysis["statecode"]
    }
)


# ============================================================
# 3. ESTIMATE TURNING POINTS
# ============================================================

def calculate_turning_point(variable):

    linear = model.params[f"{variable}_c"]

    quadratic = model.params[
        f"I({variable}_c ** 2)"
    ]

    if np.isclose(quadratic, 0):
        return np.nan

    centered_turning_point = (
        -linear / (2 * quadratic)
    )

    return (
        centered_turning_point +
        analysis[variable].mean()
    )


turning_points = {
    "obesity": calculate_turning_point("obesity"),
    "diabetes": calculate_turning_point("diabetes")
}


# ============================================================
# 4. CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = "results/figures"

os.makedirs(output_dir, exist_ok=True)

reference_state = analysis["statecode"].mode().iloc[0]


# ============================================================
# 5. GENERATE ADJUSTED PREDICTION FIGURES
# ============================================================

def create_adjusted_plot(variable, display_name, filename):

    observed_min = analysis[variable].min()
    observed_max = analysis[variable].max()

    x_grid = np.linspace(
        observed_min,
        observed_max,
        100
    )

    prediction_data = pd.DataFrame({
        "log_primary_care_access":
            analysis["log_primary_care_access"].mean(),

        "obesity_c": 0,

        "diabetes_c": 0,

        "uninsured_c":
            analysis["uninsured_c"].mean(),

        "unemployment_c":
            analysis["unemployment_c"].mean(),

        "statecode": reference_state
    }, index=range(len(x_grid)))

    prediction_data[f"{variable}_c"] = (
        x_grid - analysis[variable].mean()
    )

    design_matrix = build_design_matrices(
        [design_info],
        prediction_data,
        return_type="dataframe"
    )[0]

    X_prediction = np.asarray(design_matrix)

    beta = np.asarray(model.params)
    covariance = np.asarray(model.cov_params())

    predicted_log = X_prediction @ beta

    prediction_variance = np.einsum(
        "ij,jk,ik->i",
        X_prediction,
        covariance,
        X_prediction
    )

    prediction_se = np.sqrt(
        np.maximum(prediction_variance, 0)
    )

    lower_log = predicted_log - 1.96 * prediction_se
    upper_log = predicted_log + 1.96 * prediction_se

    predicted = np.exp(predicted_log)
    lower = np.exp(lower_log)
    upper = np.exp(upper_log)

    turning_point = turning_points[variable]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.fill_between(
        x_grid * 100,
        lower,
        upper,
        color="#4C91C6",
        alpha=0.20,
        label="95% confidence interval"
    )

    ax.plot(
        x_grid * 100,
        predicted,
        color="#1769AA",
        linewidth=2.8,
        label="Adjusted predicted hospital stays"
    )

    if (
        np.isfinite(turning_point)
        and observed_min <= turning_point <= observed_max
    ):
        ax.axvline(
            turning_point * 100,
            color="#E17C05",
            linestyle="--",
            linewidth=2,
            label=f"Estimated turning point: {turning_point * 100:.1f}%"
        )

        ax.scatter(
            turning_point * 100,
            np.exp(
                model.predict(
                    pd.DataFrame({
                        "log_primary_care_access": [
                            analysis["log_primary_care_access"].mean()
                        ],
                        "obesity_c": [
                            (turning_point - analysis["obesity"].mean())
                            if variable == "obesity" else 0
                        ],
                        "diabetes_c": [
                            (turning_point - analysis["diabetes"].mean())
                            if variable == "diabetes" else 0
                        ],
                        "uninsured_c": [
                            analysis["uninsured_c"].mean()
                        ],
                        "unemployment_c": [
                            analysis["unemployment_c"].mean()
                        ],
                        "statecode": [reference_state]
                    })
                ).iloc[0]
            ),
            color="#E17C05",
            s=65,
            zorder=5
        )

    ax.set_title(
        f"County {display_name} Prevalence and Hospital Stays",
        fontsize=16,
        pad=15
    )

    ax.set_xlabel(
        f"County {display_name} prevalence (%)",
        fontsize=12
    )

    ax.set_ylabel(
        "Adjusted geometric mean of hospital stays",
        fontsize=12
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="upper left",
        frameon=False
    )

    ax.text(
        0.5,
        -0.16,
        "Other predictors held at sample means; "
        f"state fixed effect held at reference state {int(reference_state)}. "
        "Observational associations, not causal effects.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="dimgray"
    )

    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"FIGURE CREATED: {filepath}")


create_adjusted_plot(
    "diabetes",
    "Diabetes",
    "diabetes_adjusted_relationship.png"
)

create_adjusted_plot(
    "obesity",
    "Obesity",
    "obesity_adjusted_relationship.png"
)


# ============================================================
# 6. SAVE FINAL ANALYSIS RESULTS
# ============================================================

summary_path = "results/final_analysis_summary.txt"

os.makedirs("results", exist_ok=True)

with open(summary_path, "w") as file:

    file.write("COUNTY HOSPITAL STAYS: FINAL ANALYSIS\n")
    file.write("====================================\n\n")

    file.write(f"County observations: {int(model.nobs)}\n")

    file.write(
        f"State clusters: {analysis['statecode'].nunique()}\n"
    )

    file.write(f"R-squared: {model.rsquared:.4f}\n\n")

    file.write("ESTIMATED TURNING POINTS\n")
    file.write("------------------------\n")

    for variable, value in turning_points.items():

        if np.isfinite(value):
            file.write(
                f"{variable.capitalize()}: {value * 100:.2f}%\n"
            )
        else:
            file.write(
                f"{variable.capitalize()}: Not estimable\n"
            )

    file.write("\nQUADRATIC TERM P-VALUES\n")
    file.write("----------------------\n")

    for variable in ["obesity", "diabetes"]:

        term = f"I({variable}_c ** 2)"

        file.write(
            f"{variable.capitalize()}: "
            f"{model.pvalues[term]:.6f}\n"
        )

    file.write("\nMODEL SPECIFICATION\n")
    file.write("-------------------\n")

    file.write(
        "Outcome: Log-transformed county hospital stays.\n"
    )

    file.write(
        "Predictors: Primary care access, obesity, diabetes, "
        "uninsurance, and unemployment.\n"
    )

    file.write(
        "Obesity and diabetes include quadratic terms.\n"
    )

    file.write(
        "State fixed effects included.\n"
    )

    file.write(
        "Standard errors clustered by state.\n"
    )

    file.write(
        "\nInterpretation: Adjusted observational associations. "
        "Turning points are model-derived estimates, not "
        "established clinical or policy thresholds.\n"
    )


# ============================================================
# 7. PRINT FINAL RESULTS
# ============================================================

print("\nFINAL ANALYSIS")
print("==============")

print(f"County observations: {int(model.nobs)}")
print(f"State clusters: {analysis['statecode'].nunique()}")
print(f"R-squared: {model.rsquared:.4f}")

print("\nTURNING POINTS")
print("---------------")

for variable, value in turning_points.items():

    if np.isfinite(value):
        print(f"{variable.capitalize()}: {value * 100:.2f}%")
    else:
        print(f"{variable.capitalize()}: Not estimable")

print("\nQUADRATIC TERM P-VALUES")
print("-----------------------")

for variable in ["obesity", "diabetes"]:

    term = f"I({variable}_c ** 2)"

    print(
        f"{variable.capitalize()}: "
        f"{model.pvalues[term]:.6f}"
    )

print("\nANALYSIS COMPLETE")
print("=================")

print(f"Figures saved in: {output_dir}")
print(f"Summary saved in: {summary_path}")

