from __future__ import annotations
from typing import Optional, Sequence, Dict, Any, List

import pandas as pd
import numpy as np
import io
import json
import uuid
import inspect
import base64

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import scipy
import scipy.stats
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway, multivariate_normal

from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder,
    MinMaxScaler, StandardScaler, RobustScaler
)
from sklearn.decomposition import FactorAnalysis

from IPython.display import HTML, display as _display


# ══════════════════════════════════════════════════════════════════════
#  DataInspector
# ══════════════════════════════════════════════════════════════════════

class DataInspector:

    def __init__(self):
        self.df                      = None
        self.numeric_df              = None
        self.categorical_df          = None
        self.categorical_normalized_df = None
        self.normalized_data_df      = None
        self.numeric_normalized_df   = None

    # ── ingestion ─────────────────────────────────────────────────────

    def upload_data(self):
        try:
            from google.colab import files as _gfiles
            raw = _gfiles.upload()
        except ImportError:
            raise EnvironmentError("Must run inside Google Colab. Use load_from_path() locally.")

        if not raw:
            print("Nothing was uploaded."); return

        fname   = next(iter(raw))
        content = raw[fname]
        _null   = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ']

        self.df = pd.read_csv(io.BytesIO(content), na_values=_null)
        self.df['count'] = 1

        for c in self.df.columns:
            trial = pd.to_numeric(self.df[c], errors='coerce')
            if not trial.isna().all():
                self.df[c] = trial

        print(f"\n✅ '{fname}' ingested — shape {self.df.shape}")

    def load_from_path(self, path: str):
        _null = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ']
        self.df = pd.read_csv(path, na_values=_null)
        self.df['count'] = 1

        for c in self.df.columns:
            trial = pd.to_numeric(self.df[c], errors='coerce')
            if not trial.isna().all():
                self.df[c] = trial

        print(f"✅ '{path}' loaded — shape {self.df.shape}")

    # ── structure & inspection ────────────────────────────────────────

    def get_summary(self):
        if self.df is None:
            print("No data loaded."); return

        n_rows, n_cols = self.df.shape
        nums = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cats = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        print("─── Dataset Overview ───────────────────────────")
        print(f"Rows : {n_rows}   Columns : {n_cols}")
        print(f"Numeric  ({len(nums)}) : {nums}")
        print(f"Categ.   ({len(cats)}) : {cats}")
        _display(self.df.head(20))

    def show_missing_data(self):
        if self.df is None:
            return

        has_blank  = (self.df == "").any(axis=1)
        has_null   = self.df.isnull().any(axis=1)
        bad_rows   = self.df[has_null | has_blank]

        if bad_rows.empty:
            print("✨ Dataset has no missing entries.")
        else:
            print(f"🔍 {len(bad_rows)} rows contain at least one missing value:")
            _display(bad_rows)

    def column_details(self):
        if self.df is None:
            return

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                lo, hi = self.df[col].min(), self.df[col].max()
                print(f"🔹 {col} [numeric]  range: {lo} → {hi}")
            else:
                n_unique = self.df[col].nunique()
                print(f"🔸 {col} [category] distinct values: {n_unique}")

    def get_categorical_summary(self):
        if self.df is None:
            return

        cat_frame = self.df.select_dtypes(exclude=[np.number])
        if cat_frame.empty:
            print("No categorical columns present."); return

        tbl = cat_frame.describe().T[['unique', 'top', 'freq']]
        print("─── Categorical Summary ───────────────────────")
        _display(tbl)

    # ── cleaning ──────────────────────────────────────────────────────

    def handle_missing_values(self, columns=None, strategy='median', fill_value=None):
        if self.df is None:
            return

        targets = columns if columns else \
                  self.df.columns[self.df.isnull().any()].tolist()

        for col in targets:
            if col not in self.df.columns:
                print(f"⚠️  '{col}' not found — skipped."); continue

            s = self.df[col]
            is_num = pd.api.types.is_numeric_dtype(s)

            if   strategy == 'mean'     and is_num : self.df[col] = s.fillna(s.mean())
            elif strategy == 'median'   and is_num : self.df[col] = s.fillna(s.median())
            elif strategy == 'mode'                : self.df[col] = s.fillna(s.mode().iat[0])
            elif strategy == 'constant'            :
                if fill_value is None:
                    raise ValueError("fill_value required for strategy='constant'")
                self.df[col] = s.fillna(fill_value)
            else:
                print(f"⚠️  '{strategy}' inapplicable to '{col}' — skipped.")

        print(f"🛠️  Imputation done ({strategy}) on: {targets}")

    def remove_duplicates(self):
        if self.df is None:
            return

        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        print(f"✨ {before - len(self.df)} duplicate rows removed. Total now: {len(self.df)}")

    def delete_rows(self):
        if self.df is None:
            return

        try:
            raw    = input("Row indices to remove (comma-separated): ")
            chosen = [int(x.strip()) for x in raw.split(',') if x.strip().isdigit()]
            valid  = [i for i in chosen if i in self.df.index]
            self.df = self.df.drop(index=valid).reset_index(drop=True)
            print(f"🗑️  Removed {len(valid)} rows. Remaining: {len(self.df)}")
        except Exception as exc:
            print(f"❌ {exc}")

    def delete_columns(self):
        if self.df is None:
            print("No data loaded."); return

        try:
            print("Current columns:", ', '.join(self.df.columns))
            raw    = input("Column names to drop (comma-separated): ")
            wanted = [c.strip() for c in raw.split(',')]
            found  = [c for c in wanted if c in self.df.columns]

            if not found:
                print("⚠️  None of those columns exist."); return

            self.df = self.df.drop(columns=found)
            print(f"🗑️  Dropped {len(found)} column(s). Remaining: {self.df.shape[1]}")
        except Exception as exc:
            print(f"❌ {exc}")

    def handle_outliers(self, columns=None, find_and_delete=False):
        if self.df is None:
            return

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        targets  = columns if columns else num_cols
        flagged  = set()

        for col in targets:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"⚠️  '{col}' is non-numeric — skipped."); continue

            q1, q3  = self.df[col].quantile([0.25, 0.75])
            fence   = 1.5 * (q3 - q1)
            mask    = (self.df[col] < q1 - fence) | (self.df[col] > q3 + fence)
            outlier_idx = self.df.index[mask].tolist()
            flagged.update(outlier_idx)
            print(f"🚨 {col}: {len(outlier_idx)} outlier(s) detected.")

        if flagged:
            _display(self.df.loc[sorted(flagged)])
            if find_and_delete:
                self.df = self.df.drop(index=sorted(flagged)).reset_index(drop=True)
                print(f"🗑️  {len(flagged)} outlier row(s) removed.")

    def export_cleaned_data(self, filename='cleaned_data.csv'):
        if self.df is None:
            return

        self.df.to_csv(filename, index=False)
        try:
            from google.colab import files as _gfiles
            _gfiles.download(filename)
        except ImportError:
            print(f"⚠️  Not in Colab — '{filename}' saved locally.")
        print(f"💾 Export complete: '{filename}'")

    # ── feature extraction ────────────────────────────────────────────

    def extract_numeric_data(self):
        if self.df is None:
            return print("No data loaded.")
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        return self.numeric_df

    def extract_categorical_data(self):
        if self.df is None:
            return print("No data loaded.")
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])
        return self.categorical_df

    # ── normalisation ─────────────────────────────────────────────────

    def extract_normalized_numeric_data(self, method='minmax'):
        if self.df is None:
            return print("No data loaded.")

        frame = self.df.select_dtypes(include=[np.number]).copy()
        if frame.empty:
            print("⚠️  No numeric columns."); self.numeric_normalized_df = pd.DataFrame(); return self.numeric_normalized_df

        if frame.isnull().values.any():
            print("ℹ️  NaN detected — filling medians before scaling.")
            frame = frame.fillna(frame.median())

        _map = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}
        key  = method.lower().strip()

        if key not in _map:
            print(f"❌ Unknown method '{method}' — falling back to minmax.")
            return self.extract_normalized_numeric_data('minmax')

        scaled = _map[key].fit_transform(frame)
        self.numeric_normalized_df = pd.DataFrame(scaled, columns=frame.columns, index=frame.index)
        print(f"✨ Numeric columns scaled via '{key}'.")
        return self.numeric_normalized_df

    def extract_normalized_categorical_data(self, method='uniform'):
        if self.df is None:
            return print("No data loaded.")

        frame = self.df.select_dtypes(exclude=[np.number]).copy()
        if frame.empty:
            print("⚠️  No categorical columns."); self.categorical_normalized_df = pd.DataFrame(); return self.categorical_normalized_df

        key = method.lower().strip()

        if key == 'uniform':
            for col in frame.columns:
                codes = frame[col].astype('category').cat.codes
                mx    = codes.max()
                frame[col] = codes / mx if mx > 0 else 0.0
            self.categorical_normalized_df = frame

        elif key == 'ordinal':
            enc  = OrdinalEncoder()
            data = enc.fit_transform(frame.fillna('_missing_'))
            self.categorical_normalized_df = pd.DataFrame(data, columns=frame.columns, index=frame.index)

        elif key == 'onehot':
            enc  = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            data = enc.fit_transform(frame.fillna('_missing_'))
            cols = enc.get_feature_names_out(frame.columns)
            self.categorical_normalized_df = pd.DataFrame(data, columns=cols, index=frame.index)

        elif key == 'minmax_ordinal':
            enc    = OrdinalEncoder()
            scaler = MinMaxScaler()
            data   = scaler.fit_transform(enc.fit_transform(frame.fillna('_missing_')))
            self.categorical_normalized_df = pd.DataFrame(data, columns=frame.columns, index=frame.index)

        else:
            print(f"❌ Unknown method '{method}' — falling back to uniform.")
            return self.extract_normalized_categorical_data('uniform')

        print(f"✨ Categorical columns encoded via '{key}'.")
        return self.categorical_normalized_df

    def create_normalized_data_df(self):
        if self.df is None:
            return print("No data loaded.")

        num_part = self.extract_numeric_data()
        cat_part = self.extract_normalized_categorical_data()

        both_empty = lambda f: f is None or (isinstance(f, pd.DataFrame) and f.empty)

        if both_empty(cat_part):
            print("ℹ️  No categorical columns — returning numeric only.")
            self.normalized_data_df = num_part; return self.normalized_data_df

        if both_empty(num_part):
            print("ℹ️  No numeric columns — returning encoded categorical only.")
            self.normalized_data_df = cat_part; return self.normalized_data_df

        self.normalized_data_df = pd.concat(
            [num_part.reset_index(drop=True), cat_part.reset_index(drop=True)], axis=1
        )
        print(f"✅ Merged frame ready — {self.normalized_data_df.shape[1]} columns.")
        return self.normalized_data_df

    # ── visualisation ─────────────────────────────────────────────────

    def plot_numerical(self, column_names):
        if self.df is None:
            return

        cols = [column_names] if isinstance(column_names, str) else column_names
        cols = [c for c in cols if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]

        for col in cols:
            series = self.df[col]
            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=(f"Violin/Box · {col}",
                                                f"Scatter · {col}",
                                                f"Histogram · {col}"))

            fig.add_trace(go.Violin(x=series, box_visible=True, meanline_visible=True,
                                    name=col, orientation='h', line_color='lightseagreen'), row=1, col=1)
            fig.add_trace(go.Scatter(y=series, mode='markers',
                                     marker=dict(opacity=0.5, color='royalblue'), name=col), row=1, col=2)
            fig.add_trace(go.Histogram(x=series, name=col, marker_color='indianred'), row=1, col=3)

            fig.update_layout(height=450, title_text=f"<b>Analysis: {col}</b>",
                              showlegend=False, template="plotly_white")
            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            fig.update_xaxes(title_text="Value", row=1, col=3)
            fig.show()

    def plot_categorical(self, column_names):
        if self.df is None:
            return

        cols = [column_names] if isinstance(column_names, str) else column_names

        for col in cols:
            if col not in self.df.columns:
                print(f"⚠️  '{col}' not found."); continue

            freq = self.df[col].value_counts().reset_index()
            freq.columns = [col, 'n']
            freq['pct'] = (freq['n'] / freq['n'].sum() * 100).round(1).astype(str) + '%'

            fig = px.bar(freq, x=col, y='n', text='pct', title=f"Frequency — {col}",
                         color=col, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.show()

    def plot_relationship(self, col1, col2):
        if self.df is None:
            return

        n1 = pd.api.types.is_numeric_dtype(self.df[col1])
        n2 = pd.api.types.is_numeric_dtype(self.df[col2])

        if n1 and n2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline='ols',
                             title=f"{col1} vs {col2}")
        elif n1 != n2:
            num_c, cat_c = (col1, col2) if n1 else (col2, col1)
            fig = px.box(self.df, x=cat_c, y=num_c, points='all', color=cat_c,
                         title=f"{num_c} by {cat_c}")
        else:
            fig = px.histogram(self.df, x=col1, color=col2, barmode='group',
                               title=f"{col1} × {col2}")
        fig.show()

    def plot_numerical_correlation(self):
        if self.df is None:
            return

        corr_mat = self.df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr_mat, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r',
                        title='Pearson Correlation — Numeric Features')
        fig.show()

    def plot_categorical_correlation(self):
        if self.df is None:
            return print("No data loaded.")

        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("⚠️  No categorical columns.")

        labels = cat_df.columns.tolist()
        n      = len(labels)
        mat    = pd.DataFrame(np.zeros((n, n)), index=labels, columns=labels)

        for i in range(n):
            for j in range(i, n):
                a, b = labels[i], labels[j]
                if i == j:
                    mat.loc[a, b] = 1.0; continue

                ct   = pd.crosstab(cat_df[a], cat_df[b])
                if ct.size == 0 or min(ct.shape) <= 1:
                    mat.loc[a, b] = mat.loc[b, a] = 0.0; continue

                chi2  = chi2_contingency(ct)[0]
                total = ct.to_numpy().sum()
                v     = np.sqrt(chi2 / (total * (min(ct.shape) - 1))) if total else 0.0
                mat.loc[a, b] = mat.loc[b, a] = round(v, 4)

        print("─── Cramér's V Matrix ──────────────────────────")
        _display(mat.round(3))

        fig = px.imshow(mat, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r',
                        title="<b>Cramér's V — Categorical Association</b>",
                        labels=dict(color="Cramér's V"))
        fig.update_layout(height=max(400, n * 80), width=max(500, n * 80),
                          template='plotly_white')
        fig.show()
        return mat

    def correlate_num_to_cat(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not num_cols or not cat_cols:
            print("⚠️  Need both numeric and categorical columns."); return pd.DataFrame()

        rows = []
        for cat in cat_cols:
            for num in num_cols:
                pair  = self.df[[cat, num]].dropna()
                if pair.empty: continue

                levels = pair[cat].unique()
                if len(levels) < 2: continue

                if len(levels) == 2:
                    binary = pd.get_dummies(pair[cat], drop_first=True).iloc[:, 0]
                    r, p   = pointbiserialr(binary, pair[num])
                    rows.append({'Categorical': cat, 'Numerical': num,
                                 'Type': 'Point-Biserial', 'Correlation': round(r, 3), 'P-Value': round(p, 4)})
                else:
                    groups = [pair[pair[cat] == lv][num] for lv in levels]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) < 2: continue

                    _, p       = f_oneway(*groups)
                    gm         = pair[num].mean()
                    ss_tot     = ((pair[num] - gm) ** 2).sum()
                    ss_bet     = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
                    eta        = np.sqrt(ss_bet / ss_tot) if ss_tot > 0 else 0.0
                    rows.append({'Categorical': cat, 'Numerical': num,
                                 'Type': 'Eta (ANOVA)', 'Correlation': round(eta, 3), 'P-Value': round(p, 4)})

        return pd.DataFrame(rows)

    def plot_all_associations_heatmap(self):
        if self.df is None:
            return print("No data loaded.")

        all_cols = self.df.columns.tolist()
        n        = len(all_cols)
        mat      = pd.DataFrame(np.zeros((n, n)), index=all_cols, columns=all_cols)

        for i in range(n):
            for j in range(i, n):
                ci, cj = all_cols[i], all_cols[j]
                if i == j:
                    mat.loc[ci, cj] = 1.0; continue

                sub = self.df[[ci, cj]].dropna()
                if sub.empty: continue

                ni = pd.api.types.is_numeric_dtype(sub[ci])
                nj = pd.api.types.is_numeric_dtype(sub[cj])

                if ni and nj:
                    val = abs(sub[ci].corr(sub[cj]))

                elif not ni and not nj:
                    ct    = pd.crosstab(sub[ci], sub[cj])
                    if ct.size > 0 and min(ct.shape) > 1:
                        chi2  = chi2_contingency(ct)[0]
                        tot   = ct.to_numpy().sum()
                        val   = np.sqrt(chi2 / (tot * (min(ct.shape) - 1))) if tot else 0.0
                    else:
                        val = 0.0

                else:
                    cat_c = ci if not ni else cj
                    num_c = ci if ni else cj
                    lvls  = sub[cat_c].unique()
                    if len(lvls) > 1:
                        gs    = [sub[sub[cat_c] == lv][num_c] for lv in lvls]
                        gs    = [g for g in gs if len(g) > 0]
                        gm    = sub[num_c].mean()
                        ss_t  = ((sub[num_c] - gm) ** 2).sum()
                        ss_b  = sum(len(g) * (g.mean() - gm) ** 2 for g in gs)
                        val   = np.sqrt(ss_b / ss_t) if ss_t > 0 else 0.0
                    else:
                        val = 0.0

                mat.loc[ci, cj] = mat.loc[cj, ci] = round(val, 3)

        print("─── Unified Association Matrix ─────────────────")
        _display(mat)

        fig = px.imshow(mat, text_auto='.2f', aspect='auto',
                        color_continuous_scale='viridis',
                        title='<b>Unified Association Heatmap</b>',
                        labels=dict(color='Strength'))
        fig.update_layout(height=max(500, n * 45), width=max(600, n * 45),
                          template='plotly_white')
        fig.show()
        return mat

    # ── statistical tests ─────────────────────────────────────────────

    def test_constant_mean(self, columns=None, chunks=10):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = [columns] if isinstance(columns, str) else list(columns)

        work  = self.df[cols].dropna().reset_index(drop=True)
        n, m  = work.shape
        k     = n // chunks

        if k < m:
            raise ValueError(f"Chunk size {k} < feature count {m}. Reduce chunks.")

        work['_blk'] = np.minimum(np.arange(n) // k, chunks - 1)
        mu_g = work[cols].mean().values
        W = np.zeros((m, m)); B = np.zeros((m, m))

        for _, grp in work.groupby('_blk'):
            X  = grp[cols].values
            mu = X.mean(axis=0)
            nj = len(X)
            W += (X - mu).T @ (X - mu)
            d  = (mu - mu_g).reshape(-1, 1)
            B += nj * (d @ d.T)

        eps = 1e-6 * np.eye(m)
        sW, ldW = np.linalg.slogdet(W + eps)
        sT, ldT = np.linalg.slogdet(W + B + eps)

        if sW <= 0 or sT <= 0:
            raise np.linalg.LinAlgError("Matrices degenerate — check data.")

        log_lam  = ldW - ldT
        lam      = np.exp(log_lam)
        df_      = m * (chunks - 1)
        sf       = n - 1 - (m + chunks) / 2
        chi2_val = max(0.0, -sf * log_lam)
        p        = 1.0 - scipy.stats.chi2.cdf(chi2_val, df_)

        print(f"\n── MANOVA Mean Stability (g={chunks}, m={m}) ──")
        print(f"Wilks' Λ : {lam:.5f}")
        print(f"χ²       : {chi2_val:.4f}   df: {df_}")
        print(f"p-value  : {p:.6f}")
        print("✅ No mean drift." if p > 0.05 else "🚨 Mean drift detected.")

        return {'wilks_lambda': lam, 'chi2': chi2_val, 'p_value': p, 'df': df_}

    def test_constant_covariance(self, columns=None, chunks=5):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = [columns] if isinstance(columns, str) else list(columns)

        work = self.df[cols].dropna().reset_index(drop=True)
        n, m = work.shape
        k    = n // chunks

        if k <= m:
            raise ValueError(f"DoF/chunk ({k-1}) ≤ dimensions ({m}). Reduce chunks.")

        work['_blk'] = np.minimum(np.arange(n) // k, chunks - 1)
        eps   = 1e-6 * np.eye(m)
        S_all = []
        n_all = []
        sum_logdet = 0.0
        pooled     = np.zeros((m, m))
        total_df   = 0

        for _, grp in work.groupby('_blk'):
            X   = grp[cols].values
            nj  = len(X)
            Sj  = np.cov(X, rowvar=False, ddof=1) + eps
            dfj = nj - 1
            S_all.append(Sj); n_all.append(nj)
            pooled    += dfj * Sj
            total_df  += dfj
            sg, ld     = np.linalg.slogdet(Sj)
            if sg <= 0: raise np.linalg.LinAlgError(f"Non-PD covariance in chunk.")
            sum_logdet += dfj * ld

        pooled /= total_df
        sp_s, sp_ld = np.linalg.slogdet(pooled)
        if sp_s <= 0: raise np.linalg.LinAlgError("Pooled covariance non-PD.")

        M       = total_df * sp_ld - sum_logdet
        inv_sum = sum(1.0 / (nj - 1) for nj in n_all)
        c_fac   = (inv_sum - 1.0 / total_df) * (2 * m**2 + 3*m - 1) / (6*(m+1)*(chunks-1))
        chi2_v  = max(0.0, M * (1 - c_fac))
        df_     = int(m * (m + 1) * (chunks - 1) / 2)
        p       = 1.0 - scipy.stats.chi2.cdf(chi2_v, df_)

        print(f"\n── Box's M Covariance Test (g={chunks}, m={m}) ──")
        print(f"Box's M : {M:.4f}")
        print(f"χ²      : {chi2_v:.4f}   df: {df_}")
        print(f"p-value : {p:.6f}")
        print("✅ Covariance stable." if p > 0.001 else "🚨 Covariance drift detected.")

        return {'M': M, 'chi2': chi2_v, 'p_value': p, 'df': df_}

    def test_row_independence(self, columns=None, max_lag=None):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = [columns] if isinstance(columns, str) else list(columns)

        work = self.df[cols].dropna().values
        n, m = work.shape

        if max_lag is None:
            max_lag = int(np.ceil(np.log(n)))
        if max_lag >= n:
            raise ValueError(f"max_lag ({max_lag}) must be < n ({n}).")

        Xc  = work - work.mean(axis=0)
        eps = 1e-6 * np.eye(m)
        G0  = Xc.T @ Xc / n + eps

        try:     iG0 = np.linalg.inv(G0)
        except:  iG0 = np.linalg.pinv(G0)

        Q = 0.0
        for lag in range(1, max_lag + 1):
            Gk = Xc[lag:].T @ Xc[:-lag] / n
            Q += np.trace(Gk.T @ iG0 @ Gk @ iG0) / (n - lag)
        Q = max(0.0, Q * n**2)

        df_ = m**2 * max_lag
        p   = 1.0 - scipy.stats.chi2.cdf(Q, df_)

        print(f"\n── Multivariate Ljung-Box (lags={max_lag}) ──")
        print(f"Q_m    : {Q:.4f}   df: {df_}")
        print(f"p-value: {p:.6f}")
        print("✅ Rows independent." if p > 0.05 else "🚨 Serial dependence detected.")

        return {'Q_m': Q, 'p_value': p, 'df': df_}

    def estimate_joint_normal(self, columns=None):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = list(columns)

        X    = self.df[cols].dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("n must exceed m.")

        mu  = X.mean(axis=0)
        S   = np.cov(X, rowvar=False, ddof=1) + 1e-6 * np.eye(m)
        dist = multivariate_normal(mean=mu, cov=S, allow_singular=True)
        ll   = dist.logpdf(X).sum()
        k    = m + m * (m + 1) // 2
        aic  = 2*k - 2*ll

        print(f"\n── Joint Normal Fit  m={m}, n={n} ──")
        for c, v in zip(cols, mu): print(f"  μ[{c}] = {v:.4f}")
        print(f"Log-Likelihood: {ll:.4f}   AIC: {aic:.4f}")

        return {'mean_vector': mu, 'covariance_matrix': S,
                'log_likelihood': ll, 'aic': aic,
                'distribution_object': dist, 'features': cols}

    def instantiate_macro_clt_distribution(self, columns=None):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = list(columns)

        X    = self.df[cols].dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("n must exceed m.")

        mu   = X.mean(axis=0)
        S    = np.cov(X, rowvar=False, ddof=1)
        S_clt = S / n + 1e-10 * np.eye(m)
        dist  = multivariate_normal(mean=mu, cov=S_clt, allow_singular=True)
        tr    = np.trace(S_clt)

        print(f"\n── CLT Sampling Distribution  n={n}, m={m} ──")
        for c, v in zip(cols, mu): print(f"  μ̂[{c}] = {v:.4f}")
        print(f"Tr[(1/n)S] = {tr:.8f}")

        return {'mean_vector': mu, 'clt_covariance_matrix': S_clt,
                'total_parameter_variance': tr,
                'distribution_object': dist, 'features': cols}

    def compute_empirical_pca(self, columns=None, show_plot=True):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = list(columns)

        X    = self.df[cols].dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("n must exceed m.")

        mu   = X.mean(axis=0)
        Xc   = X - mu
        S    = np.cov(X, rowvar=False, ddof=1)

        evals, evecs = np.linalg.eigh(S)
        order  = np.argsort(evals)[::-1]
        lam    = np.clip(evals[order], 1e-15, None)
        P      = evecs[:, order]

        tot    = lam.sum()
        if tot == 0: raise ValueError("Zero total variance.")
        evr    = lam / tot
        cev    = np.cumsum(evr)
        uev    = 1.0 - cev

        Z      = Xc @ P
        S_Z    = np.cov(Z, rowvar=False, ddof=1)

        k_vals = np.arange(1, m)
        T2_arr, Q_arr = [], []
        T2_mat = np.zeros((n, len(k_vals)))
        Q_mat  = np.zeros((n, len(k_vals)))

        for ki, k in enumerate(k_vals):
            t2 = np.sum(Z[:, :k]**2 / lam[:k], axis=1)
            q  = np.sum(Z[:, k:]**2, axis=1)
            T2_mat[:, ki] = t2;  T2_arr.append(t2.mean())
            Q_mat[:, ki]  = q;   Q_arr.append(q.mean())

        print(f"\n── PCA  m={m}, n={n}, Tr[S]={tot:.4f} ──")

        if show_plot:
            pc_lbl = [f"PC{i+1}" for i in range(m)]
            k_lbl  = [f"k={k}" for k in k_vals]

            fig = make_subplots(rows=2, cols=3,
                                horizontal_spacing=0.18, vertical_spacing=0.28,
                                subplot_titles=("Loading Matrix |P|",
                                                "Eigenvalues λ",
                                                "Explained Variance",
                                                "Unexplained Variance",
                                                "Mean T² vs k",
                                                "Mean Q (SPE) vs k"))

            fig.add_trace(go.Heatmap(z=np.abs(P), x=pc_lbl, y=cols,
                                     colorscale='YlOrRd', showscale=True,
                                     colorbar=dict(title="Weight", x=-0.12, len=0.38,
                                                   y=0.78, yanchor="middle", xanchor="right",
                                                   titleside="top")), row=1, col=1)
            fig.update_xaxes(title_text="Principal Axes", row=1, col=1)

            fig.add_trace(go.Bar(x=pc_lbl, y=lam, name="λ",
                                 marker=dict(color='#1f77b4', line=dict(color='black', width=0.5))),
                          row=1, col=2)
            fig.update_yaxes(title_text="Variance", row=1, col=2)
            fig.update_xaxes(title_text="Principal Axes", row=1, col=2)

            fig.add_trace(go.Bar(x=pc_lbl, y=evr*100, name="Marginal",
                                 marker=dict(color='#ff7f0e', opacity=0.75)), row=1, col=3)
            fig.add_trace(go.Scatter(x=pc_lbl, y=cev*100, mode='lines+markers',
                                     name='Cumulative',
                                     line=dict(color='#d62728', width=2.5, dash='dash')), row=1, col=3)
            fig.update_yaxes(title_text="Captured (%)", range=[-2, 105], row=1, col=3)

            fig.add_trace(go.Bar(x=pc_lbl, y=uev*100, name="Noise",
                                 marker=dict(color='#2ca02c', line=dict(color='black', width=0.5))),
                          row=2, col=1)
            fig.update_yaxes(title_text="Excluded (%)", range=[-2, 105], row=2, col=1)

            fig.add_trace(go.Scatter(x=k_lbl, y=T2_arr, mode='lines+markers', name='T²',
                                     line=dict(color='#9467bd', width=2.5),
                                     marker=dict(size=6, symbol='diamond')), row=2, col=2)
            fig.update_yaxes(title_text="Mean T²", row=2, col=2)
            fig.update_xaxes(title_text="k", row=2, col=2)

            fig.add_trace(go.Scatter(x=k_lbl, y=Q_arr, mode='lines+markers', name='Q',
                                     line=dict(color='#e377c2', width=2.5),
                                     marker=dict(size=6, symbol='square')), row=2, col=3)
            fig.update_yaxes(title_text="Mean Q", row=2, col=3)
            fig.update_xaxes(title_text="k", row=2, col=3)

            fig.update_layout(
                title=dict(text="PCA Diagnostic Dashboard", x=0.5, y=0.97,
                           xanchor='center', yanchor='top'),
                template='plotly_white', showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                margin=dict(t=150, b=60, l=140, r=80), height=750, width=1250)
            fig.show()

        return {'mean_vector': mu, 'covariance_matrix_S': S,
                'eigenvalues_lambda': lam, 'eigenvectors_P': P,
                'explained_variance_ratio': evr,
                'cumulative_variance_ratio': cev,
                'unexplained_variance_ratio': uev,
                'transformed_scores_Z': Z,
                'score_covariance_diagonal': np.diag(S_Z),
                'features': cols, 'k_values': k_vals,
                'T2_matrix_vs_k': T2_mat, 'Q_matrix_vs_k': Q_mat,
                'mean_T2_profile': np.array(T2_arr),
                'mean_Q_profile': np.array(Q_arr)}

    def compute_empirical_fa(self, k, columns=None, show_plot=True):
        if self.df is None: raise ValueError("No data loaded.")

        if columns is None:
            cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != 'count']
        else:
            cols = list(columns)

        X    = self.df[cols].dropna().values
        n, m = X.shape
        if n <= m: raise ValueError("n must exceed m.")
        if k >= m: raise ValueError(f"k ({k}) must be < m ({m}).")

        mu   = X.mean(axis=0)
        std  = X.std(axis=0, ddof=1)
        std[std == 0] = 1e-15
        Z    = (X - mu) / std
        R    = np.corrcoef(X, rowvar=False)

        fa   = FactorAnalysis(n_components=k, rotation='varimax', random_state=42)
        fa.fit(Z)

        Lam  = fa.components_.T
        psi  = fa.noise_variance_
        comm = (Lam**2).sum(axis=1)
        F    = fa.transform(Z)

        print(f"\n── Factor Analysis  k={k}, m={m}, n={n} ──")
        print(f"Avg communality : {comm.mean()*100:.2f}%")
        print(f"Avg uniqueness  : {psi.mean()*100:.2f}%")

        if show_plot:
            f_lbl = [f"F{j+1}" for j in range(k)]

            fig = make_subplots(rows=2, cols=2,
                                horizontal_spacing=0.24, vertical_spacing=0.28,
                                subplot_titles=("Loadings |Λ|",
                                                "Communality vs Uniqueness",
                                                "Uniqueness Profile φ²",
                                                "Factor Score Variance"))

            fig.add_trace(go.Heatmap(z=np.abs(Lam), x=f_lbl, y=cols,
                                     colorscale='YlOrRd', showscale=True,
                                     colorbar=dict(title="Sensitivity", x=-0.15, len=0.38,
                                                   y=0.78, yanchor="middle", xanchor="right",
                                                   titleside="top")), row=1, col=1)
            fig.update_xaxes(title_text="Latent Factors", row=1, col=1)

            fig.add_trace(go.Bar(y=cols, x=comm*100, name="Communality",
                                 orientation='h', marker=dict(color='#1f77b4')), row=1, col=2)
            fig.add_trace(go.Bar(y=cols, x=psi*100, name="Uniqueness",
                                 orientation='h', marker=dict(color='#ff7f0e')), row=1, col=2)
            fig.update_layout(barmode='stack')
            fig.update_xaxes(title_text="Variance (%)", range=[0, 100], row=1, col=2)

            fig.add_trace(go.Scatter(x=cols, y=psi, mode='lines+markers',
                                     name='φ²', line=dict(color='#d62728', width=2, dash='dot'),
                                     marker=dict(size=8, symbol='x')), row=2, col=1)
            fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
            fig.update_xaxes(title_text="Channels", tickangle=25, row=2, col=1)

            fv = np.var(F, axis=0, ddof=1)
            fig.add_trace(go.Bar(x=f_lbl, y=fv, name="Score Variance",
                                 marker=dict(color='#2ca02c', line=dict(color='black', width=0.5))),
                          row=2, col=2)
            fig.update_yaxes(title_text="Variance", row=2, col=2)
            fig.update_xaxes(title_text="Factors", row=2, col=2)

            fig.update_layout(
                title=dict(text="Factor Analysis Dashboard", x=0.5, y=0.97,
                           xanchor='center', yanchor='top'),
                template='plotly_white', showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                margin=dict(t=150, b=60, l=140, r=80), height=750, width=1250)
            fig.show()

        return {'mean_vector_mu': mu, 'std_vector_D': std,
                'correlation_matrix_R': R,
                'factor_loadings_lambda': Lam, 'uniqueness_psi': psi,
                'communality_h2': comm, 'latent_factor_scores_F': F,
                'sensors': cols}


# ══════════════════════════════════════════════════════════════════════
#  PlottingMethods
# ══════════════════════════════════════════════════════════════════════

class PlottingMethods:

    # ── internal helpers ──────────────────────────────────────────────

    def get_methods_info(self, user_id=None):
        entries = []
        for name, fn in inspect.getmembers(self, inspect.ismethod):
            if name.startswith('_'): continue
            entries.append({
                'method':      name,
                'signature':   str(inspect.signature(fn)),
                'description': (fn.__doc__ or '').strip() or 'No description.'
            })
        return {'status': 'success', 'response': entries}

    def _data_validate(self, data, msg):
        empty = lambda: {'status': 'error', 'message_dict': msg}

        if data is None or data == '':
            msg['message'] = 'No data'; return empty()

        if isinstance(data, pd.DataFrame):
            if data.empty: msg['message'] = 'No data'; return empty()
            return {'status': 'success', 'data': data.to_dict(orient='records')}

        if isinstance(data, list):
            if not data: msg['message'] = 'No data'; return empty()
            return {'status': 'success', 'data': data}

        try:
            parsed  = json.loads(data)
            records = parsed.get('records') if isinstance(parsed, dict) else parsed
            if not records: msg['message'] = 'No data'; return empty()
            return {'status': 'success', 'data': records}
        except Exception:
            msg['message'] = 'Invalid data format'; return empty()

    def _wrap_fig(self, fig):
        html   = pio.to_html(fig, full_html=False, include_plotlyjs=True,
                             config={'displaylogo': False, 'responsive': True})
        div_id = str(uuid.uuid4())[:8]
        return html.replace('<div>', f'<div id="{div_id}">', 1)

    def _err(self, msg_dict, msg_text):
        msg_dict['message'] = msg_text
        return {'status': 'error',
                'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': ''})},
                'message': msg_text}

    def display_image(self, result):
        if result['status'] == 'success':
            _display(HTML(json.loads(result['response']['data'])['figure']))
        else:
            print(f"Plot failed: {result.get('message', 'unknown error')}")

    # ── charts ────────────────────────────────────────────────────────

    def plot_bar_chart(self, x='date', y='value', color=None, text=None,
                       title='', barmode='stack', hover_data=None,
                       data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            records = vr['data']

            if isinstance(hover_data, str):
                try:    hover_data = json.loads(hover_data)
                except: hover_data = [h.strip() for h in hover_data.split(',')] if ',' in hover_data else None

            df = pd.DataFrame(records)
            df[y] = pd.to_numeric(df[y], errors='coerce')

            c_cats = None
            if color and color in df.columns:
                df.dropna(subset=[color], inplace=True)
                c_cats = sorted(df[color].unique())
                df[color] = pd.Categorical(df[color], categories=c_cats, ordered=True)

            x_cats = df[x].unique()
            df[x]  = pd.Categorical(df[x], categories=x_cats, ordered=True)

            if hover_data:
                hover_data = [c for c in hover_data if c in df.columns]

            cat_ord = {x: x_cats}
            if color and c_cats is not None: cat_ord[color] = c_cats

            fig = px.bar(df, x=x, y=y, color=color, title=title, text=text,
                         hover_data=hover_data, category_orders=cat_ord)
            fig.update_layout(xaxis_title=x, yaxis_title=y,
                              uniformtext_minsize=8, uniformtext_mode='hide', barmode=barmode)

            md['message'] = 'Bar chart plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_pie_chart(self, names='date', values='value', title='', hole=None,
                       data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df  = pd.DataFrame(vr['data'])
            fig = px.pie(df, names=names, values=values, title=title, hole=hole)
            fig.update_traces(textinfo='percent+label')

            md['message'] = 'Pie chart plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_histogram(self, x='value', title='', bins=None,
                       data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df = pd.DataFrame(vr['data'])

            if bins:
                if not isinstance(bins, list) or len(bins) < 2:
                    return self._err(md, 'bins must be a list with at least 2 edges.')
                df[x] = pd.cut(df[x], bins=bins, right=False).astype(str)

            fig = px.histogram(df, x=x, title=title)
            fig.update_layout(xaxis_title=x, yaxis_title='Count', bargap=0.2)

            md['message'] = 'Histogram plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_simple_sunburst_graph(self, path=["parent", "name"], values="marks",
                                   title='Hierarchy map', data_id=None,
                                   data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df  = pd.DataFrame(vr['data']).fillna('')
            fig = px.sunburst(df, path=path, values=values, title=title)

            md['message'] = 'Sunburst plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_tree_map(self, path=["parent", "name"], values="marks",
                      title='Hierarchy map', data_id=None,
                      data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df  = pd.DataFrame(vr['data'])
            fig = px.treemap(df, path=path, values=values, title=title)

            md['message'] = 'Treemap plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_sankey_diagram(self, source_column="parent", target_column="name",
                            values="marks", title="Sankey Diagram",
                            data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df      = pd.DataFrame(vr['data'])
            agg     = df.groupby([source_column, target_column], as_index=False).agg({values: 'sum'})
            nodes   = pd.concat([agg[source_column], agg[target_column]]).unique()
            idx_map = {nd: i for i, nd in enumerate(nodes)}

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5),
                          label=list(idx_map.keys())),
                link=dict(source=agg[source_column].map(idx_map).tolist(),
                          target=agg[target_column].map(idx_map).tolist(),
                          value=agg[values].tolist()))])
            fig.update_layout(title_text=title, font_size=10)

            md['message'] = 'Sankey diagram plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_sunburst_from_hierarchy(self, path=["parent", "name"], values="marks",
                                     title="Sunburst Diagram", data_id=None,
                                     data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            raw_df    = pd.DataFrame(vr['data']).fillna('')
            root_name = 'Root'

            # identify top-level node
            all_parents  = set(raw_df[path[0]].replace('', root_name))
            all_children = set(raw_df[path[1]])
            top_nodes    = all_parents - all_children
            root         = top_nodes.pop() if len(top_nodes) == 1 else root_name

            # accumulate values bottom-up
            val_map = {row[path[1]]: row[values]
                       for _, row in raw_df.iterrows() if pd.notna(row[values])}

            def _accum(node, parent_map, vmap):
                if node not in parent_map: return vmap.get(node, 0)
                total = sum(_accum(ch, parent_map, vmap) for ch in parent_map[node])
                vmap[node] = total; return total

            p_map = {}
            for _, row in raw_df.iterrows():
                p = row[path[0]] or root_name
                p_map.setdefault(p, []).append(row[path[1]])
            _accum(root, p_map, val_map)

            # build flat rows for px.sunburst
            def _walk(node, parent, pm, vm, out):
                out.append({'id': node, 'parent': parent, 'value': vm.get(node, 0)})
                for ch in pm.get(node, []):
                    _walk(ch, node, pm, vm, out)

            rows = []
            _walk(root, '', p_map, val_map, rows)
            hdf = pd.DataFrame(rows)

            fig = go.Figure(go.Sunburst(ids=hdf['id'], labels=hdf['id'],
                                        parents=hdf['parent'], values=hdf['value'],
                                        branchvalues='total'))
            fig.update_layout(title=title)

            md['message'] = 'Hierarchy sunburst plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_heat_map(self, values='Sales', index='Region', columns='Category',
                      aggregade_method='sum', fill_value=0,
                      title='Heatmap', width=None, data_id=None,
                      data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df   = pd.DataFrame(vr['data'])
            rows = df[index].unique()
            cols = df[columns].unique()

            pivot = df.pivot_table(index=index, columns=columns, values=values,
                                   aggfunc=aggregade_method,
                                   fill_value=fill_value).reindex(index=rows, columns=cols)

            fig = px.imshow(pivot, color_continuous_scale='Jet', text_auto=True, title=title,
                            labels=dict(y=index, x=columns, color=values))
            fig.update_layout(autosize=True, width=width)

            md['message'] = 'Heatmap plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_multi_column_bar_graph(self, xLabel="Week",
                                    value_vars=['Metric A', 'Metric B'],
                                    title="Multi-column Bar", hover_data=[],
                                    barmode='group', data_id=None,
                                    data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            vr = self._data_validate(data, md)
            if vr['status'] != 'success': return self._err(md, md.get('message', ''))

            df  = pd.DataFrame(vr['data'])
            needed = list(set([xLabel] + hover_data + value_vars))
            df  = df[[c for c in needed if c in df.columns]]

            melted = df.melt(id_vars=[xLabel] + [h for h in hover_data if h in df.columns],
                             value_vars=[v for v in value_vars if v in df.columns],
                             var_name='Series', value_name='Value')

            fig = px.bar(melted, x=xLabel, y='Value', color='Series',
                         barmode=barmode, title=title,
                         hover_data=hover_data if hover_data else None)
            fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), autosize=True,
                              legend=dict(orientation='h', x=0.5, xanchor='center',
                                          y=1, yanchor='bottom'))

            md['message'] = 'Multi-column bar plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_flow_chart(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            import graphviz

            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data

            records = parsed.get('records', {})
            if not records: return self._err(md, 'No data')

            edges     = records.get('edges', [])
            node_defs = records.get('nodes', [])

            dot = graphviz.Digraph(format='png')

            for nd in node_defs:
                lbl = nd.get('label', '')
                dot.node(lbl, label=lbl,
                         shape=nd.get('shape', 'ellipse'),
                         style=nd.get('style', 'filled'),
                         fillcolor=nd.get('fillcolor', '#bbbbbb'),
                         fontcolor=nd.get('fontcolor', 'black'))

            for ed in edges:
                dot.edge(ed.get('start', ''), ed.get('end', ''),
                         label=ed.get('label', ''),
                         color=ed.get('color', 'black'),
                         penwidth=str(ed.get('penwidth', 1)))

            png_bytes = dot.pipe(format='png')
            b64       = base64.b64encode(png_bytes).decode('utf-8')
            fig_id    = str(uuid.uuid4())[:8]
            html_img  = f'<div id="{fig_id}"><figure><img src="data:image/png;base64,{b64}" alt="flowchart"></figure></div>'

            md['message'] = 'Flowchart plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': html_img}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')

    def plot_flow_chart_plotly(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        md = {'message': meta_data}
        try:
            import networkx as nx

            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data

            records   = parsed.get('records', {})
            if not records: return self._err(md, 'No data')

            edges     = records.get('edges', [])
            node_defs = records.get('nodes', [])

            G          = nx.MultiDiGraph()
            edge_labels = {}

            for ed in edges:
                s = ed['start'].split(':')[0]
                t = ed['end'].split(':')[0]
                G.add_edge(s, t)
                edge_labels[(s, t)] = ed.get('label', '')

            for nd in node_defs:
                lbl = nd['label']
                if lbl in G.nodes:
                    G.nodes[lbl].update(nd)

            pos = nx.spring_layout(G, seed=42)

            # edge traces
            ex, ey = [], []
            for s, t in G.edges():
                x0, y0 = pos[s]; x1, y1 = pos[t]
                ex += [x0, x1, None]; ey += [y0, y1, None]
            edge_tr = go.Scatter(x=ex, y=ey, mode='lines',
                                 line=dict(width=2, color='black'), hoverinfo='none')

            # node traces
            nx_, ny_, nt_, nc_ = [], [], [], []
            for nd in G.nodes():
                x, y = pos[nd]
                nx_.append(x); ny_.append(y)
                info = G.nodes[nd]
                nt_.append(info.get('label', nd))
                nc_.append(info.get('fillcolor', '#BBBBBB'))

            node_tr = go.Scatter(x=nx_, y=ny_, mode='markers+text',
                                 marker=dict(size=40, color=nc_, line=dict(width=2, color='black')),
                                 text=nt_, textposition='middle center', hoverinfo='text')

            # edge label traces
            elx = [(pos[s][0] + pos[t][0]) / 2 for s, t in G.edges()]
            ely = [(pos[s][1] + pos[t][1]) / 2 for s, t in G.edges()]
            elt = [edge_labels.get((s, t), '') for s, t in G.edges()]
            lbl_tr = go.Scatter(x=elx, y=ely, mode='text', text=elt,
                                textposition='top center', hoverinfo='none', showlegend=False)

            fig = go.Figure(data=[edge_tr, node_tr, lbl_tr],
                            layout=go.Layout(
                                showlegend=False, hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor='white'))

            md['message'] = 'Plotly flowchart plotted'
            return {'status': 'success',
                    'response': {'meta_data': md, 'data': json.dumps({'figure': self._wrap_fig(fig)}),
                                 'message': json.dumps(md)}}
        except Exception as exc:
            return self._err(md, f'Error: {exc}')
