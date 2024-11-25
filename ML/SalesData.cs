using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SalesPrediction.ML
{
    public class SalesData
    {
        [LoadColumn(0)]
        public float Year { get; set; }

        [LoadColumn(1)]
        public string Month { get; set; }

        [LoadColumn(2)]
        public float Sales { get; set; }
    }
}