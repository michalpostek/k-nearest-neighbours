using Iris;
using KNearestNeighbours;

var data = IrisDataReader.ReadData();
var normalizedData = TrainingDataNormalizer.Normalize(data);
var distanceMetrics = new List<Func<double[], double[], double>>
{
    DistanceMetrics.ChebyshevDistance,
    DistanceMetrics.ManhattanDistance,
    DistanceMetrics.EuclideanDistance,
    DistanceMetrics.LogarithmicDistance
};
var results = new List<(string metric, int k, double precision)>();

for (var k = 1; k < 20; k++)
{
    distanceMetrics.ForEach(metric =>
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
    });
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