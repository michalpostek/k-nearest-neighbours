using Iris;
using KNearestNeighbours;

var data = IrisDataReader.ReadData();
var normalizedData = TrainingDataNormalizer.Normalize(data);

const int kFrom = 1;
const int kTo = 20;
const int minkowskiP = 3;

var distanceMetrics = new Dictionary<string, Func<double[], double[], double>>
{
    { nameof(DistanceMetrics.MinkowskiDistance), (double[] from, double[] to) => DistanceMetrics.MinkowskiDistance(from, to, minkowskiP) },
    { nameof(DistanceMetrics.ChebyshevDistance), DistanceMetrics.ChebyshevDistance },
    { nameof(DistanceMetrics.ManhattanDistance), DistanceMetrics.ManhattanDistance },
    { nameof(DistanceMetrics.EuclideanDistance), DistanceMetrics.ChebyshevDistance },
    { nameof(DistanceMetrics.LogarithmicDistance), DistanceMetrics.LogarithmicDistance },
};
var results = new List<(string metric, int k, double precision)>();

for (var k = kFrom; k < kTo; k++)
{
    foreach (var metric in distanceMetrics.Values)
    {
        var incorrectClassifications = 0;

        for (var i = 0; i < normalizedData.Count; i++)
        {
            var currentSampleIndex = i;
            var currentSample = normalizedData[currentSampleIndex];
            var currentTrainingData = normalizedData.Where((_, index) => index != currentSampleIndex).ToList();

            var result = KNearestNeighbours<IrisClass>.Classify(currentSample.Item1, currentTrainingData, k, metric);

            if (result != currentSample.Item2)
            {
                ++incorrectClassifications;
            }
        }
        
        results.Add((metric.Method.Name, k, 1 - incorrectClassifications / (double)normalizedData.Count));
    }
}

results.Sort((x, y) => y.precision.CompareTo(x.precision));

Console.WriteLine($"{"Metryka",-30}{"k",-5}{"Dokładność",-15}");
Console.WriteLine(new string('-', 50));

results.ForEach(result =>
{
    Console.WriteLine(
        $"{result.metric,-30}" +
        $"{result.k,-5}" +
        $"{Math.Round(result.precision * 100, 2) + "%",-15}"
    );
});