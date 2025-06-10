from app import app, response, limiter
from app.controller import aicontroller
from flask import request, render_template


@app.route('/api/predict', methods=["POST"])
@limiter.limit("5 per minute")  # Batasan per pengguna
def predict() : 
    if request.method == 'POST':
        return aicontroller.predict()
    else:
        return response.badRequest(None, 'Terjadi kesalahan')
    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/subdistrict', methods=['GET'])
def get_subdistrict():
    list_sub = ['Gondokusuman', 'Depok', 'Wirobrajan', 'Danurejan', 'Mergangsan',
       'Kecamatan Depok', 'Tegalrejo', 'Kecamatan Umbulharjo',
       'Umbulharjo', 'Pakualaman', 'Kecamatan Kotagede', 'Gedong Tengen',
       'Mlati', 'Ngemplak', 'Mantrijeron', 'Kecamatan Ngemplak',
       'Ngaglik', 'Kecamatan Jetis', 'Kecamatan Danurejan',
       'Kecamatan Gondokusuman', 'Kecamatan Mlati', 'Kecamatan Tegalrejo',
       'Kecamatan Ngaglik', 'Ngampilan', 'Jetis', 'Kecamatan Mergangsan',
       'Kotagede', 'Kecamatan Kraton', 'Kecamatan Mantrijeron',
       'Gondomanan', 'Kecamatan Gondomanan', 'Grogol Petamburan']
    return response.ok(list_sub, "success")