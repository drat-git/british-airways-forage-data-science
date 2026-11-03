"""Raw-input validation and loading for the selected booking model."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import cloudpickle
import numpy as np
import pandas as pd

RAW_COLUMNS = ['num_passengers', 'sales_channel', 'trip_type', 'purchase_lead',
               'length_of_stay', 'flight_hour', 'flight_day', 'route', 'booking_origin',
               'wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals', 'flight_duration']
CATEGORICAL = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin', 'airport_1', 'airport_2']
NUMERIC = [name for name in RAW_COLUMNS if name not in CATEGORICAL]

def add_airport_endpoints(frame):
    """Select the raw schema in a stable order; derive neutral route endpoints."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Provide a pandas DataFrame with the 13 raw input columns.')
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    if frame.columns.duplicated().any():
        raise ValueError('Input column names must be unique.')
    result = frame.loc[:, RAW_COLUMNS].copy()
    if result.isna().any().any():
        raise ValueError('Missing inputs are unsupported by this fitted model.')
    for name in CATEGORICAL:
        if name in result and not result[name].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f'{name} must contain strings.')
    if not result['route'].str.fullmatch(r'[A-Z]{6}').all():
        raise ValueError('route must contain two three-letter uppercase airport codes.')
    for name in NUMERIC:
        if not pd.api.types.is_numeric_dtype(result[name]) or not np.isfinite(result[name]).all():
            raise ValueError(f'{name} must contain finite numeric values.')
    for name in ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']:
        if not result[name].isin([0, 1]).all():
            raise ValueError(f'{name} must contain 0 or 1.')
    if not result['flight_hour'].between(0, 23).all():
        raise ValueError('flight_hour must be between 0 and 23.')
    if not result['flight_day'].isin(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']).all():
        raise ValueError('flight_day must use Mon through Sun labels.')
    if (result[['purchase_lead', 'length_of_stay']] < 0).any().any() or (result['num_passengers'] < 1).any() or (result['flight_duration'] <= 0).any():
        raise ValueError('Invalid passenger count, duration, lead time or length of stay.')
    result['airport_1'] = result['route'].str[:3]
    result['airport_2'] = result['route'].str[3:]
    return result

def load_selected(model_dir=None):
    """Load the local trusted bundle after checking versions and recorded hashes."""
    directory = Path(model_dir) if model_dir else Path(__file__).parent / 'model' / 'selected'
    manifest = json.loads((directory / 'manifest.json').read_text())
    for package, expected in manifest['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'Use the pinned project environment: {package}=={expected}')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['inference_code_sha256']:
        raise RuntimeError('Inference implementation differs from the exported model manifest.')
    path = directory / 'catboost_bundle.pkl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['assets_sha256'][path.name]:
        raise RuntimeError('Selected model bundle hash mismatch.')
    with path.open('rb') as stream:
        bundle = cloudpickle.load(stream)
    if bundle['threshold'] != manifest['threshold']:
        raise RuntimeError('Bundle and manifest thresholds differ.')
    return bundle

def predict_completed(new_rows, bundle=None):
    """Apply the selected CatBoost and its development-selected >= cutoff."""
    bundle = load_selected() if bundle is None else bundle
    probability = bundle['pipeline'].predict_proba(new_rows)[:, 1]
    return pd.DataFrame({'completion_probability': probability,
                         'predicted_complete': (probability >= bundle['threshold']).astype(int)}, index=new_rows.index)
