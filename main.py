from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from utils import get_unique_filters, load_data
import pandas as pd
import os

app = FastAPI(title="Dynamic Sales Forecasting API")

class ForecastRequest(BaseModel):
    filters: Optional[Dict[str, List[str]]] = None
    periods: Optional[int] = 30

@app.get("/filters")
def filters():
    return get_unique_filters()

@app.post("/forecast")
def forecast(request: ForecastRequest):
    from utils import filter_sales_data, forecast_sales
    df = load_data()
    filtered_df = filter_sales_data(df, request.filters or {})
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the given filters.")
    forecast_df = forecast_sales(filtered_df, request.periods)
    return forecast_df.to_dict(orient="records")

@app.post("/plot")
def plot(request: ForecastRequest):
    from utils import filter_sales_data, prophet_plot
    df = load_data()
    filtered_df = filter_sales_data(df, request.filters or {})
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the given filters.")
    plot_b64 = prophet_plot(filtered_df, request.periods)
    return {"plot_base64": plot_b64}

@app.post("/components")
def components(request: ForecastRequest):
    from utils import filter_sales_data, prophet_components
    df = load_data()
    filtered_df = filter_sales_data(df, request.filters or {})
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the given filters.")
    comp_b64 = prophet_components(filtered_df, request.periods)
    return {"components_base64": comp_b64}

@app.get("/plots")
def get_saved_plots():
    # List all saved plot files (base64)
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        return []
    files = [f for f in os.listdir(plot_dir) if f.endswith(".b64")]
    plots = []
    for fname in files:
        with open(os.path.join(plot_dir, fname), "r") as f:
            plots.append({"filename": fname, "base64": f.read()})
    return plots

@app.get("/all-plots", response_class=HTMLResponse)
def all_plots():
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        return "<h2>No plots found.</h2>"
    files = [f for f in os.listdir(plot_dir) if f.endswith(".b64")]
    if not files:
        return "<h2>No plots found.</h2>"
    images_html = ""
    for fname in files:
        with open(os.path.join(plot_dir, fname), "r") as f:
            b64 = f.read()
            images_html += f'<div style="margin-bottom:2rem;"><h4>{fname}</h4><img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ccc;" /></div>'
    return f"<html><head><title>All Prophet Plots</title></head><body><h2>All Prophet Plots (base64, dpi=60)</h2>{images_html}</body></html>"

@app.get("/plot-img/{filename}")
def get_plot_img(filename: str):
    plot_dir = "plots"
    file_path = os.path.join(plot_dir, filename)
    if not os.path.exists(file_path):
        return Response(status_code=404)
    with open(file_path, "r") as f:
        b64 = f.read()
    return Response(content=b64, media_type="text/plain")

@app.get("/all-plots-img", response_class=HTMLResponse)
def all_plots_img():
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        return "<h2>No plots found.</h2>"
    files = [f for f in os.listdir(plot_dir) if f.endswith(".b64")]
    if not files:
        return "<h2>No plots found.</h2>"
    images_html = ""
    for fname in files:
        images_html += f'<div style="margin-bottom:2rem;"><h4>{fname}</h4><img src="/plot-img/{fname}" style="max-width:100%;border:1px solid #ccc;" /></div>'
    return f"<html><head><title>All Prophet Plots (Image Tag)</title></head><body><h2>All Prophet Plots (as images, dpi=60)</h2>{images_html}</body></html>"

@app.get("/all-plots-img-tab")
def all_plots_img_tab():
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        return {"urls": []}
    files = [f for f in os.listdir(plot_dir) if f.endswith(".b64")]
    urls = [f"/plot-img-view/{fname}" for fname in files]
    # Return a simple HTML page with links that open each plot in a new tab
    links_html = "<h2>Open Prophet Plots in New Tabs</h2><ul>"
    for url, fname in zip(urls, files):
        links_html += f'<li><a href="{url}" target="_blank">{fname}</a></li>'
    links_html += "</ul>"
    return HTMLResponse(content=links_html)

@app.get("/plot-img-view/{filename}", response_class=HTMLResponse)
def plot_img_view(filename: str):
    plot_dir = "plots"
    file_path = os.path.join(plot_dir, filename)
    if not os.path.exists(file_path):
        return HTMLResponse("<h2>Plot not found.</h2>", status_code=404)
    with open(file_path, "r") as f:
        b64 = f.read()
    # Render just the image in a blank page
    return f'<html><head><title>{filename}</title></head><body style="margin:0;"><img src="data:image/png;base64,{b64}" style="display:block;max-width:100vw;max-height:100vh;margin:auto;" /></body></html>'

# Serve the frontend.html as index
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("frontend.html", "r") as f:
        return f.read()

# Optionally serve static files if needed in the future
# app.mount("/static", StaticFiles(directory="static"), name="static")
