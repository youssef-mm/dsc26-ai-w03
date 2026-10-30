# AI usage

## Tools I used, and for what

I used Gemini 3.8 Flash as an interactive pair programmer for drafting pipeline schemas, reasoning through temporal tie edge-cases, and verifying scikit-learn metric calculations.

## AI-suggested code I rejected or fixed in add_history_features or time_split

- **`add_history_features` tie-breaking flaw:**  
  The initial AI suggestion used a simple `df.sort_values("created_at").groupby("customer_id")` with `.cumcount()` and `.cumsum() - df["failed"]`. I rejected this because whenever a customer places multiple orders within the same minute, `cumcount()` sequentially increments across rows, violating the rule that same-timestamp parcels must not see each other. Grader test `history_ties_are_not_prior` caught this flaw. I fixed it by aggregating by `["customer_id", "created_at"]` first and subtracting each timestamp group's total count and failure sum, guaranteeing that tied parcels only see parcels strictly prior in time.

- **`time_split` boundary tie leakage:**  
  The AI suggested positional slicing `df.iloc[:-n_test]` and `df.iloc[-n_test:]`. I rejected this because when rows at the cut-off boundary share the same timestamp, positional slicing arbitrarily splits that timestamp between train and test. Grader test `time_split_boundary_ties` requires all parcels sharing the boundary timestamp to be assigned to the test split. I fixed it by identifying `boundary = sorted_times.iloc[-n_test]` and partitioning with `df[df[time_col] < boundary]` for train and `df[df[time_col] >= boundary]` for test.

- **`build_pipeline` missing text & small batch fragility:**  
  The initial suggestion passed `'address_text'` into `TfidfVectorizer(min_df=5)` without null handling. This crashed with `ValueError: np.nan is an invalid document` when encountering `None`/`NaN` in test inputs, and failed on small training samples where document count was less than `min_df`. I wrapped the text transformer with a `FunctionTransformer` to cast nulls to empty strings and set `min_df=1` so single-row and production edge-cases never crash.

## How I checked that my model has no leak

I verified that all post-event columns (`courier_notes`, `delivered_at`, `failed`, `parcel_id`) are excluded from `FEATURE_COLUMNS`, that out-of-fold ROC AUC is an honest ~0.63 (strictly below the 0.90 leakage threshold), and that all transformations are fitted solely on training data inside the pipeline.
