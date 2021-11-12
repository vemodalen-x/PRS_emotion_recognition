import paddlex as pdx
test_jpg = r'C:\Users\vemodalen\Desktop\PRS PROJECT\CK+48\anger\S010_004_00000017.png'
predictor = pdx.deploy.Predictor('inference_model_mobilenetv3/inference_model')
result = predictor.predict(test_jpg)
print("Predict Result: ", result[0]['category'],result[0]['score'])