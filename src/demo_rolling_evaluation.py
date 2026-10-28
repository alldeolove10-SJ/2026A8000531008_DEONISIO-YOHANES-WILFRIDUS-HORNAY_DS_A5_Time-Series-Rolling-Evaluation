"""
DS-A5 supporting example (NOT the assessed dataset/solution).
Synthetic daily demand series with a deliberately injected shift.
Run from any directory: python src/demo_rolling_evaluation.py
"""
from pathlib import Path
import hashlib, json, platform, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
for d in ["data","results","figures"]:
    (ROOT/d).mkdir(exist_ok=True)
DATA = ROOT/"data"/"synthetic_daily_demand_demo.csv"
if not DATA.exists():
    rng = np.random.default_rng(2026)
    n=720; dates=pd.date_range("2024-01-01",periods=n,freq="D"); t=np.arange(n)
    weekly=12*np.sin(2*np.pi*t/7); annualish=18*np.sin(2*np.pi*t/180)
    trend=.025*t; shift=np.where(t>=480,18.,0.); scale=np.where(t>=480,7.5,3.5)
    y=100+weekly+annualish+trend+shift+rng.normal(0,scale)
    pd.DataFrame({"date":dates,"demand_units":y}).to_csv(DATA,index=False,float_format="%.6f")
raw=pd.read_csv(DATA,parse_dates=["date"])
series=raw["demand_units"].to_numpy(dtype=float); dates=raw["date"].to_numpy()
sha=hashlib.sha256(DATA.read_bytes()).hexdigest()
assert raw["date"].is_monotonic_increasing and raw["date"].is_unique
assert raw["demand_units"].notna().all()

def features_at(indices):
    # All rolling windows end at t-1; target y[t] is not in its own predictors.
    out=[]
    for i in indices:
        assert i >= 14
        out.append({"lag_1":series[i-1],"lag_2":series[i-2],"lag_7":series[i-7],
                    "lag_14":series[i-14],"rolling_mean_7":series[i-7:i].mean(),
                    "rolling_std_7":series[i-7:i].std(ddof=1),
                    "day_of_week":pd.Timestamp(dates[i]).dayofweek})
    return pd.DataFrame(out,index=indices)

origins=[360,420,480,540,600,660]; horizon=30
metrics=[]; preds=[]
for fold,origin in enumerate(origins,1):
    test_idx=np.arange(origin,min(origin+horizon,len(series)))
    train_idx=np.arange(14,origin)
    assert train_idx.max() < test_idx.min(), "Temporal leakage: training target overlaps test period."
    assert len(test_idx)>0
    X_train=features_at(train_idx); X_test=features_at(test_idx)
    y_train=series[train_idx]; y_test=series[test_idx]
    # Pipeline is refit independently inside every rolling window.
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()),
                    ("ridge",Ridge(alpha=10.0))])
    model.fit(X_train,y_train)
    predictions={
        "Persistence":series[test_idx-1],
        "Seasonal naive (7d)":series[test_idx-7],
        "Ridge lag model":model.predict(X_test)
    }
    for i,idx in enumerate(test_idx):
        row={"fold":fold,"origin_date":str(pd.Timestamp(dates[origin]).date()),
             "target_date":str(pd.Timestamp(dates[idx]).date()),"target_index":int(idx),
             "actual":float(y_test[i])}
        for name,p in predictions.items():
            key={"Persistence":"persistence","Seasonal naive (7d)":"seasonal_naive_7","Ridge lag model":"ridge_lag_model"}[name]
            row[key]=float(p[i])
        preds.append(row)
    for name,p in predictions.items():
        metrics.append({"fold":fold,"origin_index":origin,"origin_date":str(pd.Timestamp(dates[origin]).date()),
                        "test_start":str(pd.Timestamp(dates[test_idx[0]]).date()),
                        "test_end":str(pd.Timestamp(dates[test_idx[-1]]).date()),
                        "n_train":len(train_idx),"n_test":len(test_idx),"model":name,
                        "MAE":mean_absolute_error(y_test,p),
                        "RMSE":mean_squared_error(y_test,p)**.5})
met=pd.DataFrame(metrics); pred=pd.DataFrame(preds)
met.to_csv(ROOT/"results"/"demo_rolling_metrics.csv",index=False)
pred.to_csv(ROOT/"results"/"demo_rolling_predictions.csv",index=False)

# Recreate figures deterministically; colors are easy to change.
GREEN="#16803c"; PINK="#ec168c"; DARKGREEN="#006b2d"; GRAY="#777777"
plt.rcParams.update({"font.size":9,"axes.titlesize":12,"axes.labelsize":10,"legend.fontsize":8})
fig,ax=plt.subplots(figsize=(9,4.5))
ax.plot(raw.date,series,color=GREEN,lw=1,label="Synthetic daily demand")
ax.axvline(pd.Timestamp(dates[480]),color=PINK,ls="--",label="Injected shift (demo only)")
ax.set(title="Worked Example: Synthetic Demand with a Deliberate Shift",xlabel="Date",ylabel="Demand (units)")
ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ROOT/"figures"/"01_demo_series_and_shift.png",dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5))
for name,c in [("Persistence",GREEN),("Seasonal naive (7d)",PINK),("Ridge lag model",DARKGREEN)]:
    s=met[met.model==name]; ax.plot(s.fold,s.MAE,marker="o",lw=2,color=c,label=name)
ax.set(title="MAE by Rolling Origin (Synthetic Demo)",xlabel="Rolling fold / origin order",ylabel="MAE (lower is better)")
ax.set_xticks(range(1,len(origins)+1)); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
fig.savefig(ROOT/"figures"/"02_demo_mae_by_origin.png",dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
for name,c,key in [("Persistence",GREEN,"persistence"),("Seasonal naive (7d)",PINK,"seasonal_naive_7"),("Ridge lag model",DARKGREEN,"ridge_lag_model")]:
    p=pred.sort_values("target_date")
    ax.plot(pd.to_datetime(p.target_date),np.abs(p.actual-p[key]),color=c,lw=1,label=name)
ax.set(title="Absolute Forecast Error Over Time (Synthetic Demo)",xlabel="Target date",ylabel="Absolute error")
ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ROOT/"figures"/"03_demo_error_over_time.png",dpi=220); plt.close(fig)
sample=pred[(pd.to_datetime(pred.target_date)>=pd.Timestamp(dates[450]))&(pd.to_datetime(pred.target_date)<=pd.Timestamp(dates[570]))]
fig,ax=plt.subplots(figsize=(9,4.5))
ax.plot(pd.to_datetime(sample.target_date),sample.actual,color="#222222",lw=1.5,label="Actual")
for key,name,c in [("persistence","Persistence",GREEN),("seasonal_naive_7","Seasonal naive (7d)",PINK),("ridge_lag_model","Ridge lag model",DARKGREEN)]:
    ax.plot(pd.to_datetime(sample.target_date),sample[key],color=c,lw=1,label=name)
ax.axvline(pd.Timestamp(dates[480]),color=PINK,ls="--",alpha=.8,label="Injected shift")
ax.set(title="Forecasts Around the Synthetic Shift",xlabel="Target date",ylabel="Demand (units)")
ax.legend(ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ROOT/"figures"/"04_demo_predictions_near_shift.png",dpi=220); plt.close(fig)
pred["abs_error_ridge"]=np.abs(pred.actual-pred.ridge_lag_model)
pred["period"]=np.where(pd.to_datetime(pred.target_date)<pd.Timestamp(dates[480]),"before_shift","after_shift")
fig,ax=plt.subplots(figsize=(7.5,4.5))
ax.boxplot([pred.loc[pred.period=="before_shift","abs_error_ridge"],pred.loc[pred.period=="after_shift","abs_error_ridge"]],
           tick_labels=["Before shift","After shift"])
ax.set(title="Ridge Absolute Error by Period (Synthetic Demo)",ylabel="Absolute error")
ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(ROOT/"figures"/"05_demo_error_shift_comparison.png",dpi=220); plt.close(fig)

env={"purpose":"separate synthetic worked example, not DS-A5 final answer",
     "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
     "pandas":pd.__version__,"scikit_learn":sklearn.__version__,
     "seed":2026,"rolling_origins":origins,"horizon":horizon,"dataset_sha256":sha}
(ROOT/"results"/"demo_environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
print("Synthetic demonstration completed.")
print("CSV SHA-256:",sha)
print(met.groupby("model")[["MAE","RMSE"]].mean().round(3).to_string())
print("Reminder: synthetic results demonstrate the method only; do not submit them as your own dataset findings.")
