import pandas as pd
import pickle
from flask import request
from app import response, app
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def validate_input(data):
    errors = []

    # Validasi tipe data dan nilai
    if not isinstance(data["building_year"], int):
        errors.append({"field": "building_year", "error": "Tahun kos dibangun harus berupa integer."})
    if not isinstance(data["panjang"], float):
        errors.append({"field": "panjang", "error": "Panjang kos harus berupa float."})
    if not isinstance(data["lebar"], float):
        errors.append({"field": "lebar", "error": "Lebar kos harus berupa float."})
    if not isinstance(data["deposit"], int):
        errors.append({"field": "deposit", "error": "Deposit kos harus berupa integer."})
    if data["area_subdistrict"] not in ['Gondokusuman', 'Depok', 'Wirobrajan', 'Danurejan', 'Mergangsan',
       'Kecamatan Depok', 'Tegalrejo', 'Kecamatan Umbulharjo',
       'Umbulharjo', 'Pakualaman', 'Kecamatan Kotagede', 'Gedong Tengen',
       'Mlati', 'Ngemplak', 'Mantrijeron', 'Kecamatan Ngemplak',
       'Ngaglik', 'Kecamatan Jetis', 'Kecamatan Danurejan',
       'Kecamatan Gondokusuman', 'Kecamatan Mlati', 'Kecamatan Tegalrejo',
       'Kecamatan Ngaglik', 'Ngampilan', 'Jetis', 'Kecamatan Mergangsan',
       'Kotagede', 'Kecamatan Kraton', 'Kecamatan Mantrijeron',
       'Gondomanan', 'Kecamatan Gondomanan', 'Grogol Petamburan']:  # Ganti dengan daftar area yang valid
        errors.append({"field": "area_subdistrict", "error": "Area subdistrict tidak valid."})
    if not isinstance(data["ac"], bool):
        errors.append({"field": "ac", "error": "AC harus berupa boolean."})
    if not isinstance(data["air panas"], bool):
        errors.append({"field": "air panas", "error": "Air panas harus berupa boolean."})
    if not isinstance(data["k. mandi dalam"], bool):
        errors.append({"field": "k. mandi dalam", "error": "Kamar mandi dalam harus berupa boolean."})
    if not isinstance(data["k. mandi luar.1"], bool):
        errors.append({"field": "k. mandi luar.1", "error": "Kamar mandi luar harus berupa boolean."})
    if not isinstance(data["shower"], bool):
        errors.append({"field": "shower", "error": "Shower harus berupa boolean."})
    if not isinstance(data["parkir mobil"], bool):
        errors.append({"field": "parkir mobil", "error": "Parkir mobil harus berupa boolean."})
    if not isinstance(data["wastafel.1"], bool):
        errors.append({"field": "wastafel.1", "error": "Wastafel harus berupa boolean."})
    if not isinstance(data["tv"], bool):
        errors.append({"field": "tv", "error": "TV harus berupa boolean."})
    if not isinstance(data["tv.1"], bool):
        errors.append({"field": "tv.1", "error": "TV harus berupa boolean."})
    
    # if not isinstance(data["mileage"], (int, float)) or data["mileage"] < 0:
    #     errors.append({"field": "mileage", "error": "Mileage harus berupa angka non-negatif."})
    # if data["fuelType"] not in ["Petrol", "Diesel", "Electric", "Hybrid"]:
    #     errors.append({"field": "fuelType", "error": "FuelType harus berupa salah satu dari: Petrol, Diesel, Electric, Hybrid."})
    # if not isinstance(data["tax"], (int, float)) or data["tax"] < 0:
    #     errors.append({"field": "tax", "error": "Tax harus berupa angka non-negatif."})
    # if not isinstance(data["mpg"], (int, float)) or data["mpg"] < 0:
    #     errors.append({"field": "mpg", "error": "MPG harus berupa angka non-negatif."})
    # if not isinstance(data["engineSize"], (int, float)) or data["engineSize"] <= 0:
    #     errors.append({"field": "engineSize", "error": "EngineSize harus berupa angka positif."})

    return errors

def get_data():
    try:
        
        data = {
        'ac': request.json['ac'], # boolean
        'air panas':request.json['air_panas'], # boolean
        'k. mandi dalam':request.json['k_mandi_dalam'], # boolean
        'lebar':request.json['lebar'], # 1-10 done
        'parkir mobil':request.json['parkir_mobil'], # boolean
        'panjang':request.json['panjang'], # 1-10 done
        'building_year':request.json['building_year'], # tahun done
        'shower':request.json['shower'], # boolean
        'deposit':request.json['deposit'], # 0-750000 done
        'wastafel.1':request.json['wastafel.1'], # boolean
        'tv':request.json['tv'], # boolean
        'area_subdistrict':request.json['area_subdistrict'], # string done 
        'tv.1':request.json['tv.1'], # boolean
        'k. mandi luar.1':request.json['k_mandi_luar.1'], # boolean
        'Unnamed: 0' : 0,
        'maks_orang' : 0,
        'denda_keterlambatan' : 0
        }
        data['lebar'] = float(data['lebar'])
        data['panjang'] = float(data['panjang'])
        validation_errors = validate_input(data)
        if validation_errors:
            print(validation_errors)
            return False, response.badRequest(validation_errors, "Validasi gagal")
        
        input_df = pd.DataFrame([data])
        with open("app/controller/dataset/model_random_forest.pkl", "rb") as file:
            model_random_forest = pickle.load(file)
        with open("app/controller/dataset/encoders.pkl", "rb") as file:
            encoder = pickle.load(file)

        minmax = encoder['scaler']
        label_encoder = encoder['label_encoder']
        input_df[['Unnamed: 0', 'panjang', 'lebar', 'maks_orang', 'denda_keterlambatan', 'deposit']] = \
            minmax.transform(input_df[['Unnamed: 0', 'panjang', 'lebar', 'maks_orang', 'denda_keterlambatan', 'deposit']])
        
        input_df['area_subdistrict'] = label_encoder.transform(input_df['area_subdistrict'])
        input_df = input_df.drop(columns=['Unnamed: 0', 'maks_orang', 'denda_keterlambatan'])
        return True, (model_random_forest, input_df)
    except ValueError as e:
        return False, response.badRequest('Fail', f'Terjadi kesalahan pada konversi data: {e}')
    except Exception as e:
        return False, response.badRequest('Fail', f'Gagal load model atau encoder: {e}')





def predict():
    try :
        success, result = get_data()
        if not success:
            return result
        model, input_df = result
        prediction = model["model"].predict(input_df)
        nilai = prediction[0]
        nilai_akhir = nilai - model['rmse']
        # prediksi_harga = f"Nilai: Rp{nilai_akhir:,.0f} - Rp{nilai:,.0f}"
        return response.ok({"nilai_akhir":nilai_akhir, "nilai_predict":nilai}, {'status':'success'})
    except Exception as e:
        print("Error:", e)
        return response.badRequest(None, 'Terjadi kesalahan saat memproses data')    
