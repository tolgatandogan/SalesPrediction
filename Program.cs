using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using SalesPrediction.ML;

namespace SalesPrediction
{
    internal class Program
    {
        // Veri yolu
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "data.csv");

        private static void Main(string[] args)
        {
            // ML.NET ortamını başlat
            MLContext mlContext = new MLContext();

            // Veriyi yükle
            IDataView dataView = mlContext.Data.LoadFromTextFile<SalesData>(
                path: _dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Veriyi hazırlama (özellik mühendisliği)
            var dataPipeline = mlContext.Transforms.CopyColumns(
                    outputColumnName: "Label", inputColumnName: nameof(SalesData.Sales))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "MonthEncoded", inputColumnName: nameof(SalesData.Month)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(
                    outputColumnName: "YearNormalized", inputColumnName: nameof(SalesData.Year)))
                .Append(mlContext.Transforms.Concatenate("Features", "MonthEncoded", "YearNormalized"))
                .Append(mlContext.Regression.Trainers.FastTree());

            // Modeli eğit
            Console.WriteLine("Model eğitiliyor...");
            var model = dataPipeline.Fit(dataView);

            // Modeli test et
            Console.WriteLine("Model test ediliyor...");
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"RSquared: {metrics.RSquared:0.##}");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:0.##}");

            // Gelecekteki bir ayın tahmini
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SalesData, SalesPredictionData>(model);

            // Tahmin yapmak istediğiniz veri
            var futureData = new SalesData { Year = 2024, Month = "01" }; // 2024 Ocak ayı
            var prediction = predictionEngine.Predict(futureData);

            Console.WriteLine($"Tahmin edilen satış (2024-01): {prediction.Sales:0.##}");
        }
    }
}