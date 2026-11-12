import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import patsy
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

data_path = "data/raw/analytic_supplement_20260325[1].csv"
output_dir = "results/diagnostics"
os.makedirs(output_dir, exist_ok=True)

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
# 2. FIT LINEAR AND QUADRATIC MODELS
# ============================================================

linear_formula = """
log_hospital_stays ~
log_primary_care_access +
obesity_c +
diabetes_c +
uninsured_c +
unemployment_c +
C(statecode)
"""

quadratic_formula = """
log_hospital_stays ~
log_primary_care_access +
obesity_c + I(obesity_c ** 2) +
diabetes_c + I(diabetes_c ** 2) +
uninsured_c +
unemployment_c +
C(statecode)
"""

groups = analysis["statecode"]

linear_model = smf.ols(
    linear_formula,
    data=analysis
).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups}
)

model = smf.ols(
    quadratic_formula,
    data=analysis
).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups}
)

# ============================================================
# 3. JOINT TEST OF QUADRATIC TERMS
# ============================================================

parameter_names = list(model.params.index)

restriction = np.zeros((2, len(parameter_names)))

restriction[0, parameter_names.index(
    "I(obesity_c ** 2)"
)] = 1

restriction[1, parameter_names.index(
    "I(diabetes_c ** 2)"
)] = 1

joint_test = model.wald_test(
    restriction,
    use_f=False,
    scalar=True
)

# ============================================================
# 4. TURNING POINTS AND UNCERTAINTY
# ============================================================

def turning_point(variable):
    linear = model.params[f"{variable}_c"]
    quadratic = model.params[f"I({variable}_c ** 2)"]

    return (
        analysis[variable].mean()
        - linear / (2 * quadratic)
    )

turning_points = {
    "obesity": turning_point("obesity"),
    "diabetes": turning_point("diabetes")
}

# ============================================================
# 5. RESIDUAL DIAGNOSTICS
# ============================================================

fitted = model.fittedvalues
residuals = model.resid

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(fitted, residuals, alpha=0.35, s=15)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Residuals vs. fitted values")
ax.set_xlabel("Fitted log hospital stays")
ax.set_ylabel("Residuals")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(
    f"{output_dir}/residuals_vs_fitted.png",
    dpi=200
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Normal Q-Q plot of residuals")
fig.tight_layout()
fig.savefig(
    f"{output_dir}/residual_qq_plot.png",
    dpi=200
)
plt.close(fig)

# ============================================================
# 6. INFLUENTIAL OBSERVATIONS
# ============================================================

# Classical OLS influence diagnostics are exploratory.
# They do not incorporate the clustered covariance estimator.

ordinary_model = smf.ols(
    quadratic_formula,
    data=analysis
).fit()

influence = ordinary_model.get_influence()

influence_data = analysis[
    ["statecode", "hospital_stays", "obesity", "diabetes"]
].copy()

influence_data["cooks_distance"] = (
    influence.cooks_distance[0]
)

influence_data["leverage"] = (
    influence.hat_matrix_diag
)

influence_data["studentized_residual"] = (
    influence.resid_studentized_external
)

influence_data = influence_data.sort_values(
    "cooks_distance",
    ascending=False
)

influence_data.to_csv(
    f"{output_dir}/influential_observations.csv",
    index=False
)

# ============================================================
# 7. LEAVE-ONE-STATE-OUT SENSITIVITY ANALYSIS
# ============================================================

sensitivity_results = []

for state in sorted(analysis["statecode"].unique()):

    subset = analysis[
        analysis["statecode"] != state
    ].copy()

    try:
        sensitivity_model = smf.ols(
            quadratic_formula,
            data=subset
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": subset["statecode"]}
        )

        result = {"excluded_state": state}

        for variable in ["obesity", "diabetes"]:
            linear = sensitivity_model.params[
                f"{variable}_c"
            ]

            quadratic = sensitivity_model.params[
                f"I({variable}_c ** 2)"
            ]

            result[f"{variable}_quadratic"] = quadratic

            result[f"{variable}_turning_point"] = (
                subset[variable].mean()
                - linear / (2 * quadratic)
                if abs(quadratic) > 1e-12
                else np.nan
            )

        sensitivity_results.append(result)

    except Exception as error:
        print(f"Warning: State {state} failed: {error}")

sensitivity = pd.DataFrame(sensitivity_results)

sensitivity.to_csv(
    f"{output_dir}/leave_one_state_out.csv",
    index=False
)

# ============================================================
# 8. WRITE SUMMARY
# ============================================================

with open(
    f"{output_dir}/diagnostic_summary.txt",
    "w"
) as file:

    file.write("MODEL DIAGNOSTIC SUMMARY\n")
    file.write("========================\n\n")

    file.write(f"Observations: {int(model.nobs)}\n")
    file.write(
        f"State clusters: {analysis['statecode'].nunique()}\n"
    )
    file.write(f"R-squared: {model.rsquared:.4f}\n")
    file.write(
        f"Adjusted R-squared: {model.rsquared_adj:.4f}\n\n"
    )

    file.write("INDIVIDUAL QUADRATIC TERM P-VALUES\n")
    file.write("---------------------------------\n")

    for variable in ["obesity", "diabetes"]:
        term = f"I({variable}_c ** 2)"
        file.write(
            f"{variable}: {model.pvalues[term]:.6f}\n"
        )

    file.write("\nJOINT QUADRATIC TERM TEST\n")
    file.write("------------------------\n")
    file.write(
        f"Wald chi-square: {float(joint_test.statistic):.4f}\n"
    )
    file.write(
        f"Degrees of freedom: 2\n"
    )
    file.write(
        f"P-value: {float(joint_test.pvalue):.6f}\n"
    )

    file.write("\nESTIMATED TURNING POINTS\n")
    file.write("------------------------\n")

    for variable, value in turning_points.items():
        file.write(f"{variable}: {value * 100:.2f}%\n")

    file.write("\nLEAVE-ONE-STATE-OUT SENSITIVITY\n")
    file.write("------------------------------\n")

    for variable in ["obesity", "diabetes"]:
        column = f"{variable}_turning_point"
        values = sensitivity[column].dropna() * 100

        file.write(f"\n{variable}:\n")
        file.write(f"Minimum: {values.min():.2f}%\n")
        file.write(f"Median: {values.median():.2f}%\n")
        file.write(f"Maximum: {values.max():.2f}%\n")

    file.write(
        "\nNote: Turning points are model-derived estimates, "
        "not established clinical or policy thresholds.\n"
    )

# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\nMODEL DIAGNOSTICS COMPLETE")
print("==========================")
print(f"Observations: {int(model.nobs)}")
print(f"State clusters: {analysis['statecode'].nunique()}")
print(f"R-squared: {model.rsquared:.4f}")

print("\nINDIVIDUAL QUADRATIC P-VALUES")
for variable in ["obesity", "diabetes"]:
    term = f"I({variable}_c ** 2)"
    print(f"{variable}: {model.pvalues[term]:.6f}")

print("\nJOINT QUADRATIC TEST")
print(f"Wald chi-square: {float(joint_test.statistic):.4f}")
print(f"P-value: {float(joint_test.pvalue):.6f}")

print("\nTURNING POINTS")
for variable, value in turning_points.items():
    print(f"{variable}: {value * 100:.2f}%")

print("\nLEAVE-ONE-STATE-OUT RESULTS")
for variable in ["obesity", "diabetes"]:
    values = (
        sensitivity[f"{variable}_turning_point"].dropna() * 100
    )
    print(
        f"{variable}: "
        f"range {values.min():.2f}% to {values.max():.2f}%"
    )

print(f"\nFiles saved in: {output_dir}")
