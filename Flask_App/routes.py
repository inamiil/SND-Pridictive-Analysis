from flask import Blueprint, render_template, jsonify
from DB.db_connect import run_query

routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/sales-data')
def sales_data():
    try:
        query = "SELECT TOP 10 * FROM FINAL_QUERY"
        data = run_query(query)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
