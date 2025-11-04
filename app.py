import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
import io
import statsmodels.api as sm
from xgboost import XGBRegressor


# App State
@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    target_col: Optional[str] = None
    feature_cols: List[str] = field(default_factory=list)
    methods_selected: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    last_uploaded_filename: Optional[str] = None


# Central default parameters
DEFAULT_PARAMS = {
    "GLOBAL": {
        "train_split_ratio": 0.8,
    },
    "SES": {
        "use_auto": True,
        "alpha": 0.3,
    },
    "DES": {
        "use_auto": True,
        "alpha": 0.3,
        "beta": 0.1,
    },
    "TES": {
        "use_auto": True,
        "alpha": 0.3,
        "beta": 0.1,
        "gamma": 0.1,
        "seasonal_periods": 12,
    },
    "Regression Tree": {
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_lags": 3,
        "tree_max_depth_vis": 3,
    },
    "Linear Regression": {
        "include_intercept": True,
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_lags": 3,
    },
}


# Initialization
def init_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
        st.session_state["app_state"].params = DEFAULT_PARAMS.copy()
    return st.session_state["app_state"]


# Data upload UI
def ui_data_upload(state: AppState) -> AppState:
    st.subheader("1) Load Data")

    

    uploaded_file = st.file_uploader(
        "Upload data file",
        type=["xlsx", "xls", "csv"],
        help="You can upload Excel (.xlsx, .xls) or CSV (.csv) files."
    )

    if uploaded_file is None:
        st.info("Please upload a file to continue.")
        st.divider()
        return state

    #clean previous results when new file is uploaded
    file_name = uploaded_file.name
    if state.last_uploaded_filename != file_name:
        state.results = {}
        state.last_uploaded_filename = file_name

    try:
        # --- File loading ---
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.lower().endswith(".csv"):
            # Try to detect delimiter automatically
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            uploaded_file.seek(0)
            sep = ";" if content.count(";") > content.count(",") else ","
            df = pd.read_csv(uploaded_file, sep=sep)
        else:
            st.error("Unsupported file format. Please upload Excel or CSV.")
            return state

        # --- basic validity check ---
        if df is None or df.empty:
            st.error("The uploaded file appears to be empty or unreadable.")
            return state
        
        # --- canonical time feature creation ---
        if "row_index" in df.columns:
            df.rename(columns={"row_index": "row_index_original"}, inplace=True)
        df["row_index"] = np.arange(len(df), dtype=int)

        # --- preview ---
        with st.expander("Show data preview"):
            st.dataframe(df.head(), width='stretch')

        if len(df) < 10:
            st.error("At least 10 data points are required for meaningful forecasts.")
            return state

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found — please upload a file with numeric data.")
            return state

        state.target_col = st.selectbox(
            "Target variable (forecast)",
            options=numeric_cols,
            index=0,
            help="Only numeric columns can be forecast.",
        )

        # --- Feature selection ---
        all_cols = list(df.columns)
        exclude_cols = [state.target_col]
        potential_features = [c for c in all_cols if c not in exclude_cols]
        prev_selection = state.feature_cols if state.feature_cols else []

        if potential_features:
            state.feature_cols = st.multiselect(
                "Feature columns (inputs)",
                options=potential_features,
                default=[f for f in prev_selection if f in potential_features],
                help="Used by those methods that forecast based on feature (e.g., Lin. Regression," \
                    " Regression Tree, XGBoost). " 
                    " Note ℹ️ The `row_index` column is auto-generated. "
                    "It represents the sort order of the dataset and can serve as  "
                    "time indicator when no explicit date or time column exists.",
                key="feature_cols_selector",
            )
            
        else:
            st.info("No additional feature columns available.")
            state.feature_cols = []

        state.df = df
        st.success("File successfully loaded and prepared.")

    except Exception as e:
        st.error(f"Error while loading file: {e}")

    st.divider()
    return state



# Ensure numeric features
def make_numerical(X: pd.DataFrame, raise_on_error: bool = False) -> pd.DataFrame:
    """
    Convert all features to numeric.
    - Datetime -> days since first date
    - Object -> numeric (coerce)
    - Drop non-numeric columns if raise_on_error=False
    """
    X = X.copy()

    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.datetime64):
            X[col] = (X[col] - X[col].min()).dt.days
        elif X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            except Exception:
                if raise_on_error:
                    raise ValueError(
                        f"Column '{col}' contains non-numeric values and cannot be converted."
                    )
                else:
                    st.warning(f"⚠️ Column '{col}' dropped (non-numeric).")
                    X = X.drop(columns=[col])

    X = X.select_dtypes(include=["number", "bool"]).astype(float)
    return X


# Exponential smoothing family (SES, DES, TES)
def run_exp_smoothing_family(
    df: pd.DataFrame,
    target_col: str,
    method: str,
    seasonal_periods: Optional[int] = None,
    use_auto: bool = True,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    split_ratio: Optional[float] = None,
):
    """
    Unified routine for SES, DES, and TES with rolling one-step-ahead forecast.
    - use_auto=True: optimize parameters
    - use_auto=False: use provided alpha/beta/gamma
    """
    y = df[target_col].astype(float)
    split_idx = int(len(y) * split_ratio)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if method == "SES":
        model = SimpleExpSmoothing(y_train, initialization_method="estimated")
    elif method == "DES":
        model = ExponentialSmoothing(y_train, trend="add", seasonal=None, initialization_method="estimated")
    elif method == "TES":
        if seasonal_periods is None or seasonal_periods >= len(y_train):
            seasonal_periods = max(2, int(len(y_train) / 4))
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    fit_kwargs = {"optimized": use_auto}
    if not use_auto:
        if alpha is not None:
            fit_kwargs["smoothing_level"] = alpha
        if beta is not None:
            fit_kwargs["smoothing_trend"] = beta
        if gamma is not None:
            fit_kwargs["smoothing_seasonal"] = gamma

    fit = model.fit(**fit_kwargs)

    alpha = fit.params.get("smoothing_level")
    beta = fit.params.get("smoothing_trend")
    gamma = fit.params.get("smoothing_seasonal")

    y_pred_test = []
    y_hist = y_train.copy()

    for t in range(len(y_test)):
        model_t = ExponentialSmoothing(
            y_hist,
            trend=fit.model.trend,
            seasonal=fit.model.seasonal,
            seasonal_periods=fit.model.seasonal_periods,
            initialization_method="estimated",
        )
        fit_t = model_t.fit(
            optimized=False,
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
        )

        y_pred_test.append(fit_t.forecast(1).iloc[0])
        y_hist = pd.concat([y_hist, y_test.iloc[t:t + 1]])

    y_pred_test = pd.Series(y_pred_test, index=y_test.index)
    y_pred_train = fit.fittedvalues

    mse_train = np.mean((y_train - y_pred_train) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)
    mask_train = y_train != 0
    mask_test = y_test != 0
    mape_train = np.mean(np.abs((y_train[mask_train] - y_pred_train[mask_train]) / y_train[mask_train])) * 100
    mape_test = np.mean(np.abs((y_test[mask_test] - y_pred_test[mask_test]) / y_test[mask_test])) * 100


    return {
        "method": method,
        "y_train_true": y_train,
        "y_train_pred": pd.Series(y_pred_train, index=y_train.index),
        "y_test_true": y_test,
        "y_test_pred": pd.Series(y_pred_test, index=y_test.index),
        "mse_train": float(mse_train),
        "mape_train": float(mape_train),
        "mse_test": float(mse_test),
        "mape_test": float(mape_test),
        "params_": {
            "use_auto": use_auto,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "seasonal_periods": seasonal_periods,
            "rolling": True,
        },
    }


# Regression Tree
def run_regression_tree(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_lags: int = 3,
    random_state: int = 42,
    split_ratio: Optional[float] = None,
    tree_max_depth_vis: int = 3, 
):
    """Train a regression tree with lagged target and optional features."""
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    df = df.copy()

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = y.shift(lag)

    feature_set = [f"lag_{i}" for i in range(1, n_lags + 1)]
    if feature_cols:
        feature_set += feature_cols

    df = df.dropna(subset=feature_set + [target_col])

    X = df[feature_set]
    X = make_numerical(X, raise_on_error=False)
    y = df[target_col]

    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mask_train = y_train != 0
    mask_test = y_test != 0
    mape_train = np.mean(np.abs((y_train[mask_train] - y_pred_train[mask_train]) / y_train[mask_train])) * 100
    mape_test = np.mean(np.abs((y_test[mask_test] - y_pred_test[mask_test]) / y_test[mask_test])) * 100
    tree_svg = None

    #tree visualization
    tree_svg = None
    try:
        depth = tree_max_depth_vis
        fig_width = max(10, (depth+2) * np.sqrt(model.get_n_leaves()))
        fig_height = max(6, depth * 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        tree.plot_tree(
            model,
            feature_names=X_train.columns.tolist(),
            filled=True,
            rounded=True,
            fontsize=11,
            max_depth=tree_max_depth_vis,
            ax=ax,
        )
        # --- shrink horizontal margins dynamically ---
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin + 0.05*(xmax - xmin), xmax - 0.05*(xmax - xmin))
        ax.set_aspect("auto", adjustable="box")
        plt.tight_layout(pad=0.1)

        

        buf = io.BytesIO()
        fig.savefig(buf, format="svg")   
        plt.close(fig)
        tree_svg = buf.getvalue().decode("utf-8")
    except Exception as e:
        print(f"Tree SVG build failed: {e}")


    return {
        "method": "Regression Tree",
        "y_train_true": y_train,
        "y_train_pred": pd.Series(y_pred_train, index=y_train.index),
        "y_test_true": y_test,
        "y_test_pred": pd.Series(y_pred_test, index=y_test.index),
        "mse_train": float(mse_train),
        "mape_train": float(mape_train),
        "mse_test": float(mse_test),
        "mape_test": float(mape_test),
        "params_": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "n_lags": n_lags,
            "features_used": feature_set,
        },
        "model": model,
        "tree_svg": tree_svg,
    }


# Linear Regression (OLS)
def run_linear_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    include_intercept: bool = True,
    split_ratio: Optional[float] = None,
):
    """Fit an OLS model and report coefficients/p-values."""
    df = df.copy()
    X = df[feature_cols] 
    X = make_numerical(X, raise_on_error=True)
    y = df[target_col].astype(float)

    split_idx = int(len(df) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if include_intercept:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train).fit()

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mask_train = y_train != 0
    mask_test = y_test != 0
    mape_train = np.mean(np.abs((y_train[mask_train] - y_pred_train[mask_train]) / y_train[mask_train])) * 100
    mape_test = np.mean(np.abs((y_test[mask_test] - y_pred_test[mask_test]) / y_test[mask_test])) * 100

    coef_table = [
        f"{var}={coef:.3f} (p={pval:.3f})"
        for var, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values)
    ]
    coef_summary = "; ".join(coef_table)

    return {
        "method": "Linear Regression",
        "y_train_true": y_train,
        "y_train_pred": pd.Series(y_pred_train, index=y_train.index),
        "y_test_true": y_test,
        "y_test_pred": pd.Series(y_pred_test, index=y_test.index),
        "mse_train": float(mse_train),
        "mape_train": float(mape_train),
        "mse_test": float(mse_test),
        "mape_test": float(mape_test),
        "params_": {
            "include_intercept": include_intercept,
            "features_used": list(X.columns),
        },
        "coef_summary": coef_summary,
    }


# XGBoost Regressor
def run_xgboost(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    n_lags: int = 3,
    random_state: int = 42,
    split_ratio: Optional[float] = None,
):
    """Train an XGBoost regressor with lagged target and optional features."""
    df = df.copy()
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float)

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = y.shift(lag)

    feature_set = [f"lag_{i}" for i in range(1, n_lags + 1)]
    if feature_cols:
        feature_set += feature_cols

    df = df.dropna(subset=feature_set + [target_col])
    X = df[feature_set]
    X = make_numerical(X, raise_on_error=False)
    y = df[target_col]

    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mask_train = y_train != 0
    mask_test = y_test != 0
    mape_train = np.mean(np.abs((y_train[mask_train] - y_pred_train[mask_train]) / y_train[mask_train])) * 100
    mape_test = np.mean(np.abs((y_test[mask_test] - y_pred_test[mask_test]) / y_test[mask_test])) * 100

    return {
        "method": "XGBoost",
        "y_train_true": y_train,
        "y_train_pred": pd.Series(y_pred_train, index=y_train.index),
        "y_test_true": y_test,
        "y_test_pred": pd.Series(y_pred_test, index=y_test.index),
        "mse_train": float(mse_train),
        "mape_train": float(mape_train),
        "mse_test": float(mse_test),
        "mape_test": float(mape_test),
        "params_": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_lags": n_lags,
            "features_used": feature_set,
        },
        "model": model,
    }


# Method selection UI
def ui_method_selection(state: AppState) -> AppState:
    st.subheader("2) Select Forecasting Methods")

    methods = [
        "SES",
        "DES",
        "TES",
        "Regression Tree",
        "Linear Regression",
        "XGBoost",
    ]
    state.methods_selected = st.multiselect(
        "Which methods should be used?", methods
    )

    if not state.methods_selected:
        st.info("Please select at least one method.")
        st.divider()
        return state

    st.divider()
    return state


# Model runner
def run_model(
    method_name: str,
    df: pd.DataFrame,
    target_col: str,
    params: dict,
    feature_cols: Optional[List[str]] = None,
    split_ratio: Optional[float] = None,
):
    """Unified training interface for all models. Defaults merged with user settings."""
    defaults = DEFAULT_PARAMS.get(method_name, {})
    merged = {**defaults, **(params or {})}

    if split_ratio is None:
        split_ratio = DEFAULT_PARAMS["GLOBAL"]["train_split_ratio"]

    effective_features = feature_cols if feature_cols else ["row_index"]

    if method_name in ["SES", "DES", "TES"]:
        return run_exp_smoothing_family(
            df=df,
            target_col=target_col,
            method=method_name,
            seasonal_periods=merged.get("seasonal_periods", None),
            use_auto=merged.get("use_auto", True),
            alpha=merged.get("alpha"),
            beta=merged.get("beta"),
            gamma=merged.get("gamma"),
            split_ratio=split_ratio,
        )

    elif method_name == "Linear Regression":
        return run_linear_regression(
            df=df,
            target_col=target_col,
            feature_cols=effective_features,
            include_intercept=merged.get("include_intercept", True),
            split_ratio=split_ratio,
        )

    elif method_name == "Regression Tree":
        res = run_regression_tree(
        df=df,
        target_col=target_col,
        feature_cols=effective_features,
        max_depth=merged.get("max_depth"),
        min_samples_split=merged.get("min_samples_split"),
        min_samples_leaf=merged.get("min_samples_leaf"),
        n_lags=merged.get("n_lags"),
        split_ratio=split_ratio,
        tree_max_depth_vis=merged.get("tree_max_depth_vis"), 
        )

        if res and "params_" in res:
            res["params_"]["show_tree"] = merged.get("show_tree", False)
            res["params_"]["tree_max_depth_vis"] = merged.get("tree_max_depth_vis", 3)

        return res

    elif method_name == "XGBoost":
        return run_xgboost(
            df=df,
            target_col=target_col,
            feature_cols=effective_features,
            n_estimators=merged.get("n_estimators"),
            max_depth=merged.get("max_depth"),
            learning_rate=merged.get("learning_rate"),
            n_lags=merged.get("n_lags"),
            split_ratio=split_ratio,
        )

    else:
        st.error(f"Unknown method: {method_name}")
        return None


# Parameter UI
def ui_param_selection(state: AppState) -> AppState:
    st.subheader("3) Set Parameters and Train Models")

    # Global settings
    defaults_global = DEFAULT_PARAMS["GLOBAL"]
    current_global = state.params.get("GLOBAL", {})
    merged_global = {**defaults_global, **current_global}

    with st.expander("General settings", expanded=False):
        merged_global["train_split_ratio"] = st.slider(
            "Train/Test split (training share)",
            min_value=0.5,
            max_value=0.95,
            value=float(merged_global["train_split_ratio"]),
            step=0.05,
            help="Share of data used for training (remainder is test).",
        )

    state.params["GLOBAL"] = merged_global
    split_ratio = merged_global["train_split_ratio"]

    st.divider()

    # Per-method parameter controls
    for method in state.methods_selected:
        if method not in DEFAULT_PARAMS:
            st.info(f"⚙️ Parameter controls for {method} are not implemented yet.")
            continue

        p = state.params.get(method, DEFAULT_PARAMS[method].copy())

        with st.expander(f"{method} – Parameters", expanded=False):
            for param_name, default_value in DEFAULT_PARAMS[method].items():
                if param_name == "use_auto":
                    p[param_name] = st.checkbox(
                        "Automatic parameter estimation (optimize α, β, γ)",
                        value=p.get(param_name, default_value),
                        key=f"checkbox_{param_name}_{method}",
                    )
                    continue

                if param_name in ["alpha", "beta", "gamma"]:
                    if not p.get("use_auto", True):
                        p[param_name] = st.slider(
                            param_name,
                            0.0,
                            1.0,
                            float(p.get(param_name, default_value)),
                            step=0.05,
                            key=f"slider_{param_name}_{method}",
                        )
                    continue

                if param_name == "seasonal_periods":
                    p[param_name] = st.number_input(
                        "Season length (e.g., 12 = months, 7 = weekdays)",
                        min_value=2,
                        value=p.get(param_name, default_value),
                        step=1,
                        key=f"number_input_{param_name}_{method}",
                    )
                    continue

                if param_name == "include_intercept":
                    p[param_name] = st.checkbox(
                        "Include intercept",
                        value=p.get(param_name, default_value),
                        key=f"checkbox_{param_name}_{method}",
                    )
                    continue

                if isinstance(default_value, int):
                    p[param_name] = st.slider(
                        param_name,
                        1,
                        50,
                        int(p.get(param_name, default_value)),
                        step=1,
                        key=f"slider_{param_name}_{method}",
                    )
            if method == "Regression Tree":
                p["show_tree"] = st.checkbox(
                    "Show regression tree after training",
                    value=p.get("show_tree", False),
                    help="Display the trained tree structure."
                )
                if p["show_tree"]:
                    p["tree_max_depth_vis"] = st.slider(
                        "Tree depth to display",
                        min_value=1,
                        max_value=int(p.get("max_depth", 5)),
                        value=min(3, int(p.get("max_depth", 5))),
                        step=1,
                        help="Maximum depth of the displayed tree (for readability)."
                    )

            state.params[method] = p

            if st.button(f"Train {method}", key=f"train_btn_{method}"):
                try:
                    res = run_model(
                        method_name=method,
                        df=state.df,
                        target_col=state.target_col,
                        params=p,
                        feature_cols=state.feature_cols,
                        split_ratio=split_ratio,
                    )
                    if res:
                        state.results.setdefault(method, []).append(res)  
                    st.success(f"{method} trained successfully.")
                except Exception as e:
                    st.error(f"Error in {method}: {e}")

    st.markdown("---")
    if st.button("💡 Compute all selected models", key="btn_all_models"):
        st.info("Starting batch training for all selected models...")
        for method in state.methods_selected:
            try:
                p = state.params.get(method, DEFAULT_PARAMS.get(method, {}))
                res = run_model(
                    method_name=method,
                    df=state.df,
                    target_col=state.target_col,
                    params=p,
                    feature_cols=state.feature_cols,
                    split_ratio=split_ratio,
                )
                if res:
                    state.results.setdefault(method, []).append(res)  
            except Exception as e:
                st.error(f"Error in {method}: {e}")
        st.success("✅ Finished training all selected models.")

    st.divider()
    return state


# Metrics & plots UI
def ui_metrics_display(state: AppState) -> AppState:
    st.subheader("4) Evaluation Metrics & Forecast Comparison")

    if not state.results:
        st.info("No results yet. Please train models first.")
        st.divider()
        return state

    global_params = state.params.get("GLOBAL", DEFAULT_PARAMS["GLOBAL"])
    split_ratio = global_params.get("train_split_ratio", DEFAULT_PARAMS["GLOBAL"]["train_split_ratio"])

    st.caption(f"Train/Test split: {int(split_ratio * 100)}% Train / {int((1 - split_ratio) * 100)}% Test")

    rows = []
    for method, res_list in state.results.items():
    # Each method may have multiple runs
        for i, res in enumerate(res_list if isinstance(res_list, list) else [res_list]):
            label = f"{method} #{i+1}" if isinstance(res_list, list) and len(res_list) > 1 else method

            params = res.get("params_", {})
            clean_params = {
                k: v for k, v in params.items()
                if not (v is None or (isinstance(v, float) and np.isnan(v)))
            }
            param_str = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
                for k, v in clean_params.items() if v is not None
            )

            coef_info = res.get("coef_summary", "")
            rows.append({
                "Method": label,
                "Train MSE": res.get("mse_train"),
                "Train MAPE": res.get("mape_train"),
                "Test MSE": res.get("mse_test"),
                "Test MAPE": res.get("mape_test"),
                "Parameters": (param_str + "; " + coef_info).strip("; "),
            })


    df_metrics = pd.DataFrame(rows).set_index("Method")
    numeric_cols = ["Train MSE", "Train MAPE", "Test MSE", "Test MAPE"]
    st.dataframe(df_metrics.style.format(subset=numeric_cols, formatter="{:.3f}"), width="stretch")

    # --- Clear results button ---
    st.divider()
    if st.button("🗑 Clear all forecast results"):
        state.results = {}
        st.success("All stored forecast results have been cleared.")
        st.rerun() 
    st.divider()

    export_data = {}
    export_data = {"row_index": state.df["row_index"]}
    export_data[state.target_col] = state.df[state.target_col]
    global_params = state.params.get("GLOBAL", DEFAULT_PARAMS["GLOBAL"])
    split_ratio = global_params.get("train_split_ratio", DEFAULT_PARAMS["GLOBAL"]["train_split_ratio"])
    split_index = int(len(state.df) * split_ratio)
    export_data["data_split"] = ["Train" if i < split_index else "Test" for i in range(len(state.df))]


    for method, res_list in state.results.items():
        for i, res in enumerate(res_list if isinstance(res_list, list) else [res_list]):
            method_label = f"{method}_{i+1}" if isinstance(res_list, list) and len(res_list) > 1 else method

            # Combine train + test predictions for full length
            if "y_train_pred" in res and "y_test_pred" in res:
                y_pred_full = pd.concat([res["y_train_pred"], res["y_test_pred"]])
                export_data[f"{method_label}_pred"] = y_pred_full.reindex(state.df.index).reset_index(drop=True)

    export_df = pd.DataFrame(export_data)
    base_name = getattr(state, "last_uploaded_filename", "forecast_results")
    base_stem = base_name.rsplit(".", 1)[0]  # remove extension if present

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False)
    excel_bytes = buffer.getvalue()

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📘 Export forecasted values as Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"{base_stem}_forecast_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        "📄 Export forecasted values as CSV (.csv)",
        data=csv_bytes,
        file_name=f"{base_stem}_forecast_results.csv",
        mime="text/csv",
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    first_res = next(iter(state.results.values()))

    if isinstance(first_res, list):
        first_res = first_res[0]

    if "y_train_true" in first_res and "y_test_true" in first_res:
        split_point = first_res["y_test_true"].index[0]
        y_all_true = pd.concat([first_res["y_train_true"], first_res["y_test_true"]])
    else:
        split_point = None
        y_all_true = first_res.get("y_true")


    if split_point is not None:
        ax.axvspan(
            split_point,
            first_res["y_test_true"].index[-1],
            color="gray",
            alpha=0.3,
            label="Test range",
        )

    if y_all_true is not None:
        ax.plot(y_all_true.index, y_all_true.values, color="black", linewidth=2, label="Actual")

    # --- Plot all forecasts ---
    for method, res_list in state.results.items():
        # normalize to list
        runs = res_list if isinstance(res_list, list) else [res_list]
        for i, res in enumerate(runs):
            # Build a clear label per run
            label = f"{method} #{i+1}" if len(runs) > 1 else method

            if "y_train_pred" in res and "y_test_pred" in res:
                y_all_pred = pd.concat([res["y_train_pred"], res["y_test_pred"]])
            elif "y_pred" in res:
                y_all_pred = res["y_pred"]
            else:
                continue

            ax.plot(y_all_pred.index, y_all_pred.values, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend(loc="upper left")

    title = "Forecast comparison of models (Train + Test)"
    if getattr(state, "last_uploaded_filename", None):
        fname = state.last_uploaded_filename
        ax.set_title(f"{title}\nDataset: {fname}", fontsize=12)
    else:
        ax.set_title(title, fontsize=13)

    st.pyplot(fig, width="stretch")
    plt.close(fig)


    st.divider()
   
    # --- Optional Regression Tree visualization ---
    if "Regression Tree" in state.results:
        runs = state.results["Regression Tree"]
        last_run = runs[-1] if isinstance(runs, list) else runs
        p = last_run.get("params_", {})
        if p.get("show_tree", False):
            svg = last_run.get("tree_svg")
            if svg:
                st.subheader("📊 Regression Tree (latest run)")
                # scrollable container; SVG remains vector-crisp
                st.components.v1.html(
                    f'<div style="width:100%;height:700px;overflow:auto;border:1px solid #ddd">{svg}</div>',
                    height=720,
                    scrolling=True,
                )
            else:
                st.warning("Tree visualization not available for this run.")


    return state


# Main App
def main():
    st.set_page_config(page_title="Forecasting Playground", layout="wide")
    st.title("📈 Forecasting Playground")

    state = init_state()
    state = ui_data_upload(state)

    if state.df is not None and state.target_col is not None:
        state = ui_method_selection(state)
        state = ui_param_selection(state)
        state = ui_metrics_display(state)
    else:
        st.info("Please upload a file and select a target variable.")


if __name__ == "__main__":
    main()
