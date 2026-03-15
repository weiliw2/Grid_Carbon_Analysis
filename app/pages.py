"""Page renderers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.scenarios import (
    build_location_comparison,
    calculate_data_center_metrics,
    apply_coal_to_solar_shift,
    get_country_row,
    get_feature_value,
)


def render_global_overview(country_data: pd.DataFrame) -> None:
    """Render the global summary page."""
    st.markdown("## Global Carbon Intensity Map")
    st.markdown(
        '<p class="section-intro">Explore country-level electricity carbon intensity, renewable share, and dominant fuel patterns across the current global dataset.</p>',
        unsafe_allow_html=True,
    )

    valid_data = country_data[country_data["carbon_intensity_gco2_kwh"].notna()].copy()
    avg_intensity = valid_data["carbon_intensity_gco2_kwh"].mean()
    median_intensity = valid_data["carbon_intensity_gco2_kwh"].median()
    cleanest = valid_data.nsmallest(1, "carbon_intensity_gco2_kwh").iloc[0]
    dirtiest = valid_data.nlargest(1, "carbon_intensity_gco2_kwh").iloc[0]
    avg_renewable = valid_data["renewable_percentage"].mean()
    high_renewable_count = int((valid_data["renewable_percentage"] >= 50).sum())
    coal_dominant_count = int((valid_data["dominant_fuel"] == "Coal").sum())
    low_intensity_count = int((valid_data["carbon_intensity_gco2_kwh"] < 200).sum())

    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-label">Current Baseline</div>
                <div class="insight-value">{avg_intensity:.0f} gCO2/kWh average</div>
                <div class="insight-meta">Median intensity is {median_intensity:.0f} gCO2/kWh across {len(valid_data)} countries with valid modeled output.</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Lower-Carbon Systems</div>
                <div class="insight-value">{low_intensity_count} countries below 200 gCO2/kWh</div>
                <div class="insight-meta">{high_renewable_count} countries exceed 50% renewable capacity in the current plant-level mix estimate.</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Structural Watchpoint</div>
                <div class="insight-value">{coal_dominant_count} coal-dominant systems</div>
                <div class="insight-meta">Coal remains the dominant installed fuel in a large share of the highest-intensity national grids.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="note-panel">
            <div class="note-title">What To Look For</div>
            <div class="note-body">
                The cleanest modeled grid in the current dataset is <strong>{cleanest['country']}</strong> at
                <strong>{cleanest['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh</strong>, while the most carbon-intensive is
                <strong>{dirtiest['country']}</strong> at <strong>{dirtiest['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh</strong>.
                Use the map to identify regional patterns, then compare the distribution and watchlist tables below to spot structural outliers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Global Average", f"{avg_intensity:.0f} gCO2/kWh")
    with col2:
        st.metric("Cleanest Grid", f"{cleanest['country']}: {cleanest['carbon_intensity_gco2_kwh']:.0f}")
    with col3:
        st.metric("Dirtiest Grid", f"{dirtiest['country']}: {dirtiest['carbon_intensity_gco2_kwh']:.0f}")
    with col4:
        st.metric("Avg Renewable %", f"{avg_renewable:.1f}%")

    col_map, col_dist = st.columns([1.8, 1], gap="large")

    with col_map:
        st.subheader("Carbon Intensity by Country")
        fig = px.choropleth(
            valid_data,
            locations="country",
            locationmode="ISO-3",
            color="carbon_intensity_gco2_kwh",
            hover_name="country",
            hover_data={
                "carbon_intensity_gco2_kwh": ":.0f",
                "renewable_percentage": ":.1f",
                "dominant_fuel": True,
                "country": False,
            },
            color_continuous_scale=[
                [0.0, "#1f5c4f"],
                [0.5, "#ddb96b"],
                [1.0, "#b85c38"],
            ],
            labels={"carbon_intensity_gco2_kwh": "Carbon Intensity (gCO2/kWh)"},
        )
        fig.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, projection_type="natural earth"),
            coloraxis_colorbar=dict(len=0.72, thickness=16),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        st.subheader("Distribution")
        distribution_fig = px.histogram(
            valid_data,
            x="carbon_intensity_gco2_kwh",
            nbins=24,
            color_discrete_sequence=["#1f5c4f"],
            labels={"carbon_intensity_gco2_kwh": "Carbon Intensity (gCO2/kWh)"},
        )
        distribution_fig.add_vline(
            x=avg_intensity,
            line_dash="dash",
            line_color="#b85c38",
            annotation_text="Average",
            annotation_position="top right",
        )
        distribution_fig.update_layout(
            height=270,
            bargap=0.08,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Countries",
        )
        st.plotly_chart(distribution_fig, use_container_width=True)

        renewable_fig = px.scatter(
            valid_data,
            x="renewable_percentage",
            y="carbon_intensity_gco2_kwh",
            hover_name="country",
            color="dominant_fuel",
            labels={
                "renewable_percentage": "Renewable Capacity Share (%)",
                "carbon_intensity_gco2_kwh": "Carbon Intensity (gCO2/kWh)",
            },
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        renewable_fig.update_layout(
            height=270,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Dominant Fuel",
        )
        st.plotly_chart(renewable_fig, use_container_width=True)

    st.subheader("Country Leaders and Watchlist")
    col1, col2 = st.columns(2)
    with col1:
        cleanest_10 = valid_data.nsmallest(10, "carbon_intensity_gco2_kwh")[
            ["country", "carbon_intensity_gco2_kwh", "renewable_percentage", "dominant_fuel"]
        ].rename(
            columns={
                "country": "Country",
                "carbon_intensity_gco2_kwh": "gCO2/kWh",
                "renewable_percentage": "Renewable %",
                "dominant_fuel": "Dominant Fuel",
            }
        )
        renewable_leaders = valid_data.nlargest(10, "renewable_percentage")[
            ["country", "renewable_percentage", "carbon_intensity_gco2_kwh", "dominant_fuel"]
        ].rename(
            columns={
                "country": "Country",
                "renewable_percentage": "Renewable %",
                "carbon_intensity_gco2_kwh": "gCO2/kWh",
                "dominant_fuel": "Dominant Fuel",
            }
        )
        st.markdown("**Lowest Carbon Intensity**")
        st.dataframe(cleanest_10, hide_index=True, use_container_width=True)
        st.markdown("**Highest Renewable Share**")
        st.dataframe(renewable_leaders, hide_index=True, use_container_width=True)

    with col2:
        dirtiest_10 = valid_data.nlargest(10, "carbon_intensity_gco2_kwh")[
            ["country", "carbon_intensity_gco2_kwh", "renewable_percentage", "dominant_fuel"]
        ].rename(
            columns={
                "country": "Country",
                "carbon_intensity_gco2_kwh": "gCO2/kWh",
                "renewable_percentage": "Renewable %",
                "dominant_fuel": "Dominant Fuel",
            }
        )
        transition_watchlist = valid_data[
            (valid_data["dominant_fuel"] == "Coal") & (valid_data["renewable_percentage"] < 25)
        ].nlargest(10, "carbon_intensity_gco2_kwh")[
            ["country", "carbon_intensity_gco2_kwh", "renewable_percentage", "dominant_fuel"]
        ].rename(
            columns={
                "country": "Country",
                "carbon_intensity_gco2_kwh": "gCO2/kWh",
                "renewable_percentage": "Renewable %",
                "dominant_fuel": "Dominant Fuel",
            }
        )
        st.markdown("**Highest Carbon Intensity**")
        st.dataframe(dirtiest_10, hide_index=True, use_container_width=True)
        st.markdown("**Transition Watchlist**")
        if transition_watchlist.empty:
            st.info("No countries met the current watchlist filter for coal dominance and low renewable share.")
        else:
            st.dataframe(transition_watchlist, hide_index=True, use_container_width=True)


def render_policy_simulator(ml_features: pd.DataFrame, model, selected_country: str) -> None:
    """Render the coal-to-solar transition simulator."""
    st.markdown("## Policy Impact Simulator")
    st.markdown(
        '<p class="section-intro">Test a simplified coal-to-solar transition and use the trained surrogate model to estimate directional changes in annual average grid intensity.</p>',
        unsafe_allow_html=True,
    )

    coal_mask = ml_features["Coal_pct"] > 5 if "Coal_pct" in ml_features.columns else pd.Series(False, index=ml_features.index)
    countries_with_coal = ml_features[coal_mask].index.tolist()

    if not countries_with_coal:
        st.warning("No countries with significant coal capacity found in the dataset.")
        return

    if selected_country in countries_with_coal:
        policy_country = selected_country
    else:
        default_index = countries_with_coal.index("USA") if "USA" in countries_with_coal else 0
        policy_country = countries_with_coal[default_index]
        st.caption(
            f"{selected_country} is not currently included in the coal-to-solar policy simulator selection, so this tab is showing {policy_country}."
        )

    baseline_features = ml_features.loc[[policy_country]].copy()
    baseline_intensity = model.predict(baseline_features)[0]
    coal_pct = get_feature_value(baseline_features, "Coal_pct")
    renewable_ratio = get_feature_value(baseline_features, "renewable_ratio")

    st.subheader(f"Current State: {policy_country}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Carbon Intensity", f"{baseline_intensity:.0f} gCO2/kWh")
    with col2:
        st.metric("Coal Percentage", f"{coal_pct:.1f}%")
    with col3:
        st.metric("Renewable Ratio", f"{renewable_ratio * 100:.1f}%")

    st.subheader("Adjust Energy Mix")
    coal_reduction = st.slider(
        "Replace Coal with Solar (%)",
        min_value=0,
        max_value=min(100, int(coal_pct)),
        value=0,
        step=5,
        help="Percentage of coal capacity to replace with solar",
    )

    scenario_features = apply_coal_to_solar_shift(baseline_features, coal_reduction)
    scenario_intensity = model.predict(scenario_features)[0]
    reduction = baseline_intensity - scenario_intensity
    reduction_pct = (reduction / baseline_intensity) * 100 if baseline_intensity else 0
    co2_saved = reduction * 100

    st.subheader("Scenario Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "New Carbon Intensity",
            f"{scenario_intensity:.0f} gCO2/kWh",
            delta=f"-{reduction:.0f} gCO2/kWh",
            delta_color="inverse",
        )
    with col2:
        st.metric("Reduction", f"{reduction_pct:.1f}%", help="Percentage reduction in carbon intensity")
    with col3:
        st.metric("CO2 Saved (Mt/year)", f"{co2_saved:.1f}", help="Assumes 100 TWh annual generation")

    comparison_df = pd.DataFrame(
        {
            "Scenario": ["Current", "With Transition"],
            "Carbon Intensity (gCO2/kWh)": [baseline_intensity, scenario_intensity],
        }
    )
    fig = px.bar(
        comparison_df,
        x="Scenario",
        y="Carbon Intensity (gCO2/kWh)",
        color="Scenario",
        color_discrete_map={"Current": "#9CA3AF", "With Transition": "#10B981"},
        title="Carbon Intensity Comparison",
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_country_analysis(country_data: pd.DataFrame, ml_features: pd.DataFrame, selected_country: str) -> None:
    """Render the country deep-dive page."""
    st.header("Country Deep Dive")
    st.markdown(
        '<p class="section-intro">Review one country at a time through the lens of carbon intensity, installed capacity, renewable share, and fuel-mix structure.</p>',
        unsafe_allow_html=True,
    )
    country_info = get_country_row(country_data, selected_country)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Carbon Intensity", f"{country_info['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh")
    with col2:
        st.metric("Total Capacity", f"{country_info['total_capacity_mw']:,.0f} MW")
    with col3:
        st.metric("Renewable %", f"{country_info['renewable_percentage']:.1f}%")
    with col4:
        st.metric("Dominant Fuel", country_info["dominant_fuel"])

    if selected_country in ml_features.index:
        st.subheader("Energy Mix Breakdown")
        fuel_cols = [column for column in ml_features.columns if column.endswith("_pct")]
        fuel_data = ml_features.loc[selected_country, fuel_cols]
        fuel_data = fuel_data[fuel_data > 0].sort_values(ascending=False)

        if not fuel_data.empty:
            fuel_df = pd.DataFrame(
                {
                    "Fuel Type": [column.replace("_pct", "") for column in fuel_data.index],
                    "Percentage": fuel_data.values,
                }
            )
            fig = px.pie(
                fuel_df,
                values="Percentage",
                names="Fuel Type",
                title=f"{selected_country} Energy Mix",
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.Greens_r,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fuel mix data available for this country")

    st.subheader("Comparison with Global Average")
    comparison_data = pd.DataFrame(
        {
            "Metric": ["Carbon Intensity", "Renewable %"],
            selected_country: [
                country_info["carbon_intensity_gco2_kwh"],
                country_info["renewable_percentage"],
            ],
            "Global Average": [
                country_data["carbon_intensity_gco2_kwh"].mean(),
                country_data["renewable_percentage"].mean(),
            ],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=selected_country,
            x=comparison_data["Metric"],
            y=comparison_data[selected_country],
            marker_color="#059669",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Global Average",
            x=comparison_data["Metric"],
            y=comparison_data["Global Average"],
            marker_color="#9CA3AF",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Country vs Global Average",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_center_calculator(country_data: pd.DataFrame, selected_country: str) -> None:
    """Render the data center emissions calculator page."""
    st.header("Data Center Carbon Cost Calculator")
    st.markdown(
        '<p class="section-intro">Estimate annual electricity demand, operational emissions, and carbon-cost exposure for a data center under different grid-intensity contexts.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Center Specifications")
        dc_power_mw = st.number_input(
            "Power Capacity (MW)",
            min_value=1,
            max_value=500,
            value=50,
            help="Total power capacity of the data center",
        )
        utilization = st.slider(
            "Average Utilization (%)",
            min_value=10,
            max_value=100,
            value=70,
            help="Percentage of capacity typically used",
        )
        pue = st.slider(
            "Power Usage Effectiveness (PUE)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="1.0 = perfect efficiency, industry average ~1.5-2.0",
        )

    with col2:
        st.subheader("Location")
        carbon_tax = st.number_input(
            "Carbon Tax ($/tonne CO2)",
            min_value=0,
            max_value=200,
            value=50,
            help="Current or proposed carbon tax rate",
        )
        st.info("EU ETS carbon price: ~EUR80-100/tonne | Singapore: S$25/tonne (rising to S$80)")

    dc_country = selected_country
    country_info = get_country_row(country_data, dc_country)
    metrics = calculate_data_center_metrics(
        carbon_intensity=country_info["carbon_intensity_gco2_kwh"],
        dc_power_mw=dc_power_mw,
        utilization=utilization,
        pue=pue,
        carbon_tax=carbon_tax,
    )

    st.subheader("Annual Carbon Footprint")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Energy Consumption", f"{metrics['annual_energy_gwh']:,.0f} GWh/year")
    with col2:
        st.metric("CO2 Emissions", f"{metrics['annual_emissions_tonnes']:,.0f} tonnes/year")
    with col3:
        st.metric(
            "Grid Intensity",
            f"{country_info['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh",
            help=f"{dc_country} grid carbon intensity",
        )
    with col4:
        st.metric(
            "Carbon Tax Cost",
            f"${metrics['annual_carbon_cost']:,.0f}/year",
            delta=f"${metrics['annual_carbon_cost'] / 12:,.0f}/month",
            help="Annual carbon tax liability",
        )

    st.subheader("Location Comparison")
    comparison_df = build_location_comparison(
        country_data,
        metrics["annual_energy_gwh"],
        carbon_tax,
        selected_country=dc_country,
    )
    fig = px.bar(
        comparison_df,
        x="Country",
        y="Annual Emissions (tonnes)",
        color="Renewable %",
        color_continuous_scale="Greens",
        title=f"Annual Emissions for {dc_power_mw}MW Data Center by Location",
        hover_data=["Carbon Intensity", "Annual Carbon Cost ($)"],
    )
    fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Potential Savings")
    if comparison_df.empty:
        st.warning("No comparison countries with valid carbon-intensity data are available for this selection.")
        return

    cleanest_location = comparison_df.iloc[0]
    current_rows = comparison_df[comparison_df["Country"] == dc_country]
    if current_rows.empty:
        st.warning(f"{dc_country} is not available in the current location comparison set.")
        return
    current_location = current_rows.iloc[0]
    emission_savings = current_location["Annual Emissions (tonnes)"] - cleanest_location["Annual Emissions (tonnes)"]
    cost_savings = current_location["Annual Carbon Cost ($)"] - cleanest_location["Annual Carbon Cost ($)"]

    if emission_savings > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Moving to {cleanest_location['Country']} could save:")
            st.markdown(
                f"""
                * **{emission_savings:,.0f} tonnes CO2/year**
                * **${cost_savings:,.0f}/year** in carbon costs
                * **{(emission_savings / current_location['Annual Emissions (tonnes)']) * 100:.1f}%** emission reduction
                """
            )
        with col2:
            ten_year_savings = cost_savings * 10
            st.info("10-Year Projection:")
            st.markdown(
                f"""
                * Total carbon cost savings: **${ten_year_savings:,.0f}**
                * Total emissions avoided: **{emission_savings * 10:,.0f} tonnes**
                * Equivalent to taking **{(emission_savings * 10 / 4.6):,.0f} cars** off the road
                """
            )
    else:
        st.success(f"{dc_country} is already among the cleanest locations for data centers!")

    st.subheader("Carbon Reduction Strategies")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**On-site Solutions**")
        st.markdown(
            """
            * Install solar panels (reduce grid dependency)
            * Battery storage for load shifting
            * Power Purchase Agreements (PPAs) for renewables
            * Improve PUE through cooling optimization
            """
        )
    with col2:
        st.markdown("**Location Strategies**")
        st.markdown(
            """
            * Prioritize regions with clean grids
            * Consider carbon intensity in site selection
            * Multi-region deployment for workload shifting
            * Partner with utilities on renewable projects
            """
        )

    with st.expander("View Detailed Comparison Table"):
        st.dataframe(
            comparison_df.style.format(
                {
                    "Carbon Intensity": "{:.0f}",
                    "Annual Emissions (tonnes)": "{:,.0f}",
                    "Annual Carbon Cost ($)": "${:,.0f}",
                    "Renewable %": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_validation(validation_data) -> None:
    """Render benchmark validation results when available."""
    st.header("Benchmark Validation")
    st.markdown(
        '<p class="section-intro">Compare modeled country-level carbon intensity against an external benchmark dataset to quantify accuracy, bias, and where the current methodology needs refinement.</p>',
        unsafe_allow_html=True,
    )

    if validation_data is None:
        st.info(
            "No benchmark dataset found yet. Add `data/benchmarks/country_carbon_intensity_benchmark.csv` "
            "to enable validation in this page."
        )
        st.markdown(
            """
            Expected columns:
            - `country`
            - `benchmark_carbon_intensity_gco2_kwh`

            Optional columns:
            - `source`
            - `year`
            - `notes`
            """
        )
        st.caption(
            "A starter template is available at "
            "`data/benchmarks/country_carbon_intensity_benchmark_template.csv`."
        )
        return

    comparison_df = validation_data["comparison"].copy()
    metrics = validation_data["metrics"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Countries Compared", f"{metrics['num_countries_compared']}")
    with col2:
        st.metric("MAE", f"{metrics['mae_gco2_kwh']:.1f} gCO2/kWh")
    with col3:
        st.metric("RMSE", f"{metrics['rmse_gco2_kwh']:.1f} gCO2/kWh")
    with col4:
        st.metric("MAPE", f"{metrics['mape_percent']:.1f}%")
    with col5:
        correlation = metrics["correlation"]
        correlation_label = "n/a" if pd.isna(correlation) else f"{correlation:.3f}"
        st.metric("Correlation", correlation_label)

    st.subheader("Parity View")
    parity_fig = px.scatter(
        comparison_df,
        x="benchmark_carbon_intensity_gco2_kwh",
        y="carbon_intensity_gco2_kwh",
        hover_name="country",
        hover_data={
            "error_gco2_kwh": ":.1f",
            "absolute_error_gco2_kwh": ":.1f",
            "benchmark_carbon_intensity_gco2_kwh": ":.1f",
            "carbon_intensity_gco2_kwh": ":.1f",
        },
        labels={
            "benchmark_carbon_intensity_gco2_kwh": "Benchmark (gCO2/kWh)",
            "carbon_intensity_gco2_kwh": "Model Output (gCO2/kWh)",
        },
        color="absolute_error_gco2_kwh",
        color_continuous_scale="YlOrRd",
        title="Model vs Benchmark Carbon Intensity",
    )
    max_axis = max(
        comparison_df["benchmark_carbon_intensity_gco2_kwh"].max(),
        comparison_df["carbon_intensity_gco2_kwh"].max(),
    )
    parity_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_axis,
        y1=max_axis,
        line=dict(color="#6B7280", dash="dash"),
    )
    parity_fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(parity_fig, use_container_width=True)

    st.subheader("Bias and Outliers")
    col1, col2 = st.columns(2)

    overestimated = comparison_df.sort_values("error_gco2_kwh", ascending=False).head(10)
    underestimated = comparison_df.sort_values("error_gco2_kwh", ascending=True).head(10)

    with col1:
        st.markdown("**Most Overestimated Countries**")
        st.dataframe(
            overestimated[
                [
                    "country",
                    "carbon_intensity_gco2_kwh",
                    "benchmark_carbon_intensity_gco2_kwh",
                    "error_gco2_kwh",
                ]
            ].rename(
                columns={
                    "country": "Country",
                    "carbon_intensity_gco2_kwh": "Model",
                    "benchmark_carbon_intensity_gco2_kwh": "Benchmark",
                    "error_gco2_kwh": "Error",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("**Most Underestimated Countries**")
        st.dataframe(
            underestimated[
                [
                    "country",
                    "carbon_intensity_gco2_kwh",
                    "benchmark_carbon_intensity_gco2_kwh",
                    "error_gco2_kwh",
                ]
            ].rename(
                columns={
                    "country": "Country",
                    "carbon_intensity_gco2_kwh": "Model",
                    "benchmark_carbon_intensity_gco2_kwh": "Benchmark",
                    "error_gco2_kwh": "Error",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Absolute Error Ranking")
    ranked_errors = comparison_df[
        [
            "country",
            "carbon_intensity_gco2_kwh",
            "benchmark_carbon_intensity_gco2_kwh",
            "absolute_error_gco2_kwh",
            "absolute_percentage_error",
        ]
    ].rename(
        columns={
            "country": "Country",
            "carbon_intensity_gco2_kwh": "Model",
            "benchmark_carbon_intensity_gco2_kwh": "Benchmark",
            "absolute_error_gco2_kwh": "Absolute Error",
            "absolute_percentage_error": "Absolute % Error",
        }
    )
    st.dataframe(
        ranked_errors.style.format(
            {
                "Model": "{:.1f}",
                "Benchmark": "{:.1f}",
                "Absolute Error": "{:.1f}",
                "Absolute % Error": "{:.1f}%",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
