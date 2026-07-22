from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            RevolvingUtilizationOfUnsecuredLines=float(request.form.get('RevolvingUtilizationOfUnsecuredLines')),
            age=int(request.form.get('age')),
            NumberOfTime30_59DaysPastDueNotWorse=int(request.form.get('NumberOfTime30_59DaysPastDueNotWorse')),
            DebtRatio=float(request.form.get('DebtRatio')),
            MonthlyIncome=float(request.form.get('MonthlyIncome')),
            NumberOfOpenCreditLinesAndLoans=int(request.form.get('NumberOfOpenCreditLinesAndLoans')),
            NumberOfTimes90DaysLate=int(request.form.get('NumberOfTimes90DaysLate')),
            NumberRealEstateLoansOrLines=int(request.form.get('NumberRealEstateLoansOrLines')),
            NumberOfTime60_89DaysPastDueNotWorse=int(request.form.get('NumberOfTime60_89DaysPastDueNotWorse')),
            NumberOfDependents=float(request.form.get('NumberOfDependents')),
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        preds, proba = predict_pipeline.predict(pred_df)

        result = "High Risk of Default" if preds[0] == 1 else "Low Risk of Default"
        probability = round(float(proba[0]) * 100, 2)

        return render_template('index.html', results=result, probability=probability)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)