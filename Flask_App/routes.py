from flask import Blueprint, render_template, jsonify, request
from DB.db_connect import run_query, get_geo_data
from DB.model.regionmodel_code import predict_sales
from DB.model.brandmodel_code import generate_sales_forecast
from DB.EDA_Geographical import get_daily_sales_by_month_for_territory
import pandas as pd

# DIVISION
from DB.FRONTEND_DIVISION_EDA import (
    get_region_sales, get_division_sales, get_yearly_sales_by_division,
    get_monthly_sales_combined, get_monthly_sales_by_division, get_heatmap_data
)

# DISTRIBUTOR
from DB.DISTRIBUTOR_EDA import (
    get_top_distributors, get_yearly_sales_for_top_distributors,
    get_division_vs_distributor_heatmap, get_stacked_sales_by_distributor_division,
    get_distributor_type_sales, get_national_top_brands, get_regional_top_brands
)

# SKU
from DB.eda_sku import (
    get_top_skus, get_pack_size_sales_by_division, get_top_brands_by_division,
    get_top_brand_packsize_combos, get_top_flavours_by_division
)

routes = Blueprint('routes', __name__)

# === FRONTEND ===
@routes.route('/')
def index():
    return render_template('indexmodel.html')  # main page

@routes.route('/regionmodel')
def model_page():
    return render_template('regionmodel.html')  # Region model page

@routes.route('/brandmodel')
def model2_page():
    return render_template('brandmodel.html')  # Brand model page

@routes.route('/api/go-to-predictions')
def go_to_predictions():
    return render_template('/brandfurther.html')


# === EXISTING API ===
@routes.route('/api/predict-forecast', methods=['POST'])
def predict_forecast():
    data = request.get_json()
    division = data.get('division', '')
    result = predict_sales(division)
    return jsonify(result)

@routes.route('/api/region-top-brands', methods=['GET'])
def region_top_brands():
    region = request.args.get('region', '')
    if not region:
        return jsonify({"error": "Region parameter is required"}), 400
    try:
        result = generate_sales_forecast(var_region=region)
        return jsonify({
            "status": "success",
            "data": {
                "summary": result["summary"],
                "actual_vs_predicted": result["actual_vs_predicted"],
                "brands": result["brands"],
                "regions": result["regions"]
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# === NEW API ENDPOINTS ===
@routes.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        region = request.args.get('region')
        forecast_data = generate_sales_forecast(var_region=region)
        response = {
            'status': 'success',
            'data': {
                'summary': forecast_data['summary'],
                'actual_vs_predicted': forecast_data['actual_vs_predicted'],
                'brands': forecast_data['brands'],
                'regions': forecast_data['regions']
            }
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@routes.route('/api/summary', methods=['GET'])
def get_summary():
    try:
        region = request.args.get('region')
        brand = request.args.get('brand')
        forecast_data = generate_sales_forecast(var_region=region)
        summary_df = pd.DataFrame(forecast_data['summary'])
        
        if brand:
            summary_df = summary_df[summary_df['BRAND'] == brand]
        
        summary_grouped = summary_df.groupby(['DATE', 'TYPE', 'REGION', 'BRAND'])['SALES'].sum().reset_index()
        summary_grouped['DATE'] = pd.to_datetime(summary_grouped['DATE']).dt.strftime('%Y-%m-%d')
        
        response = {
            'status': 'success',
            'data': summary_grouped.to_dict(orient='records')
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@routes.route('/api/actual-vs-predicted', methods=['GET'])
def get_actual_vs_predicted():
    try:
        region = request.args.get('region')
        brand = request.args.get('brand')
        forecast_data = generate_sales_forecast(var_region=region)
        avp_df = pd.DataFrame(forecast_data['actual_vs_predicted'])
        
        if brand:
            avp_df = avp_df[avp_df['BRAND'] == brand]
        
        avp_df['DATE'] = pd.to_datetime(avp_df['DATE']).dt.strftime('%Y-%m-%d')
        
        response = {
            'status': 'success',
            'data': avp_df.to_dict(orient='records')
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@routes.route('/api/brands', methods=['GET'])
def get_brands():
    try:
        region = request.args.get('region')
        forecast_data = generate_sales_forecast(var_region=region)
        response = {
            'status': 'success',
            'data': forecast_data['brands']
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@routes.route('/api/regions', methods=['GET'])
def get_regions():
    try:
        forecast_data = generate_sales_forecast()
        response = {
            'status': 'success',
            'data': forecast_data['regions']
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
# === DIVISION ===
@routes.route('/api/region-sales')
def api_region_sales():
    return jsonify(get_region_sales())

@routes.route('/api/division-sales')
def api_division_sales():
    return jsonify(get_division_sales())

@routes.route('/api/yearly-sales-by-division')
def api_yearly_sales_by_division():
    return jsonify(get_yearly_sales_by_division())

@routes.route('/api/monthly-sales-combined')
def api_monthly_sales_combined():
    return jsonify(get_monthly_sales_combined())

@routes.route('/api/monthly-sales-by-division')
def api_monthly_sales_by_division():
    return jsonify(get_monthly_sales_by_division())

@routes.route('/api/division-heatmap')
def api_division_heatmap():
    return jsonify(get_heatmap_data())

# === DISTRIBUTOR ===
@routes.route('/api/top-distributors')
def api_top_distributors():
    return jsonify(get_top_distributors())

@routes.route('/api/yearly-sales-top-distributors')
def api_yearly_sales_top_distributors():
    return jsonify(get_yearly_sales_for_top_distributors())

@routes.route('/api/distributor-division-heatmap')
def api_distributor_division_heatmap():
    return jsonify(get_division_vs_distributor_heatmap())

@routes.route('/api/stacked-sales-distributor-division')
def api_stacked_sales():
    return jsonify(get_stacked_sales_by_distributor_division())

@routes.route('/api/distributor-type-sales')
def api_distributor_type_sales():
    return jsonify(get_distributor_type_sales())

# === SKU ===
@routes.route('/api/top-skus')
def api_top_skus():
    return jsonify(get_top_skus())

@routes.route('/api/pack-size-sales-by-division')
def api_packsize_sales():
    return jsonify(get_pack_size_sales_by_division())

@routes.route('/api/top-brands-by-division')
def api_top_brands_by_division():
    return jsonify(get_top_brands_by_division())

@routes.route('/api/brand-packsize-combos')
def api_brand_packsize_combos():
    return jsonify(get_top_brand_packsize_combos())

@routes.route('/api/top-flavours-by-division')
def api_top_flavours_by_division():
    return jsonify(get_top_flavours_by_division())

# === BRANDS TAB NEW ROUTES ===
@routes.route('/api/national-top-brands')
def api_national_top_brands():
    return jsonify(get_national_top_brands())

@routes.route('/api/regional-top-brands')
def api_regional_top_brands():
    return jsonify(get_regional_top_brands())

# === Anomaly Detection (Daily Sales Per Territory Monthly) ===
@routes.route('/api/daily-sales', methods=['GET'])
def get_daily_sales_api():
    territory = request.args.get('territory')
    division = request.args.get('division')
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)

    # 🚨 Handle missing inputs
    if not all([territory, division, year, month]):
        return jsonify({"error": "Missing required parameters"}), 400

    df = get_geo_data()
    result = get_daily_sales_by_month_for_territory(df, territory, division)

    key = f"{year}-{str(month).zfill(2)}"
    return jsonify(result.get(key) or {"data": [], "top_3": [], "bottom_3": []})