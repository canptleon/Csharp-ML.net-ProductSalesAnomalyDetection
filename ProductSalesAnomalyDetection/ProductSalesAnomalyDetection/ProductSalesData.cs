using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ProductSalesAnomalyDetection
{
    public class ProductSalesData
    {
        [LoadColumn(0)]
        public string Month;

        [LoadColumn(1)]
        public float numSales;
    }

    public class ProductSalesPrediction
    {
        
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
