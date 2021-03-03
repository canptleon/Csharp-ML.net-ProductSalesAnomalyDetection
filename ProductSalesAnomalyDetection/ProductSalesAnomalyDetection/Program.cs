using System;
using System.IO;
using Microsoft.ML;
using System.Collections.Generic;

namespace ProductSalesAnomalyDetection
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");

        const int _docsize = 36;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            DetectSpike(mlContext, _docsize, dataView);

            DetectChangepoint(mlContext, _docsize, dataView);
        }
        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            Console.WriteLine("Detect temporary changes in pattern");


            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, pvalueHistoryLength: docSize / 4);
           

            Console.WriteLine("=============== Training the model ===============");

            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));


            Console.WriteLine("=============== End of training process ===============");

            IDataView transformedData = iidSpikeTransform.Transform(productSales);

            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            Console.WriteLine("Detect Persistent changes in pattern");

            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, changeHistoryLength: docSize / 4);
            

            Console.WriteLine("=============== Training the model Using Change Point Detection Algorithm===============");

            var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));

            Console.WriteLine("=============== End of training process ===============");

            IDataView transformedData = iidChangePointTransform.Transform(productSales);

            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- alert is on, predicted changepoint";
                }
                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }
    }
}
