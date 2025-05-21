using System.Globalization;
using Iris;
using KNearestNeighbours;
using Spectre.Console;

var data = IrisDataReader.ReadData();
var normalizedData = TrainingDataNormalizer.Normalize(data);

const int kFrom = 1;
const int kTo = 20;
const int minkowskiP = 3;

var distanceMetrics = new Dictionary<string, Func<double[], double[], double>>
{
    { $"{nameof(DistanceMetrics.MinkowskiDistance)} (p = {minkowskiP})", (double[] from, double[] to) => DistanceMetrics.MinkowskiDistance(from, to, minkowskiP) },
    { nameof(DistanceMetrics.ChebyshevDistance), DistanceMetrics.ChebyshevDistance },
    { nameof(DistanceMetrics.ManhattanDistance), DistanceMetrics.ManhattanDistance },
    { nameof(DistanceMetrics.EuclideanDistance), DistanceMetrics.ChebyshevDistance },
    { nameof(DistanceMetrics.LogarithmicDistance), DistanceMetrics.LogarithmicDistance },
};
var results = new List<(string metric, int k, double precision)>();

for (var k = kFrom; k < kTo; k++)
{
    foreach (var (key, metric) in distanceMetrics)
    {
        var incorrectClassifications = 0;

        for (var i = 0; i < normalizedData.Count; i++)
        {
            var currentSampleIndex = i;
            var currentSample = normalizedData[currentSampleIndex];
            var currentTrainingData = normalizedData.Where((_, index) => index != currentSampleIndex).ToList();

            var result = KNearestNeighbours<IrisLabel>.Classify(currentSample.Item1, currentTrainingData, k, metric);

            if (result != currentSample.Item2)
            {
                ++incorrectClassifications;
            }
        }
        
        results.Add((key, k, 1 - incorrectClassifications / (double)normalizedData.Count));
    }
}

var table = new Table();
table.AddColumn("Metryka");
table.AddColumn("k");
table.AddColumn("Dokładność (%)");

results.Sort((x, y) => y.precision.CompareTo(x.precision));
results.ForEach(result =>
{
    table.AddRow(
        result.metric, 
        result.k.ToString(), 
        Math.Round(result.precision * 100, 2).ToString(CultureInfo.CurrentCulture)
    );
});

AnsiConsole.Write(table);
