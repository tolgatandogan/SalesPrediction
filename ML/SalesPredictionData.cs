using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SalesPrediction.ML
{
    public class SalesPredictionData
    {
        [ColumnName("Score")]
        public float Sales { get; set; }
    }
}